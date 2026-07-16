"""Tests for θ warm-lane soft-floor proactive dispatch throttle (task 2443).

θ adds an EARLIER, PROACTIVE soft-floor admission check ahead of the
existing ε hard-floor disk-guard (task 1860): before allocating a NEW
divergent warm lane, consult reify's `warm-lane-disk-guard.sh check --soft`
(a soft floor ABOVE the hard floor). See PRD
reify/docs/prds/warm-lane-pool-sizing-lifecycle.md task θ, contract §9.5,
boundary test B10.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, WarmLaneUnavailable, WorktreeInfo, _run

# ---------------------------------------------------------------------------
# Repo fixture (mirrors test_warm_lane_disk_guard.py's git_repo fixture)
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
    """Temporary git repository with an initial commit.

    Pre-creates the default warm-lane base and pool-storage sentinel so
    tests that go through acquire_warm_lane()/create_worktree() see a
    healthy base and mounted pool storage — mirrors
    test_warm_lane_disk_guard.py's git_repo fixture verbatim (this test
    file is self-contained per the task-2443 plan).
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


# ---------------------------------------------------------------------------
# Disk-guard stub script helpers (mirrors test_warm_lane_disk_guard.py)
# ---------------------------------------------------------------------------


def _write_disk_guard_stubs(scripts_dir: Path) -> None:
    """Write a warm-lane-disk-guard.sh stub into scripts_dir.

    Appends "check <argv>" to <repo>/.test_disk_call_log, pops the next
    exit code from <repo>/.test_disk_check_exits (one per line; 0 if
    absent/empty), and exits with that code. Flags are ignored by the stub
    (it only records them) — the CLI-contract tests inspect the call log.
    """
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


async def _add_disk_guard_scripts(repo: Path) -> None:
    """Commit a stub warm-lane-disk-guard.sh into repo/scripts/."""
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    _write_disk_guard_stubs(scripts_dir)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add disk-guard stub script'], cwd=repo)


async def _add_all_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed, debug-port, and disk-guard scripts into repo/scripts/.

    Combines a seed + debug-port stub (mirrors test_git_ops._add_warm_lane_scripts)
    with the disk-guard stub so acquire_warm_lane() can be exercised end-to-end.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)

    seed = scripts_dir / 'seed-warm-lane.sh'
    seed.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho "seeded" > "$2/target/seeded.bin"\n'
    )
    seed.chmod(0o755)

    debug = scripts_dir / 'setup-worktree-debug-port.sh'
    debug.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug.chmod(0o755)

    _write_disk_guard_stubs(scripts_dir)

    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add all warm-lane stub scripts'], cwd=repo)


async def _get_head(repo: Path) -> str:
    """Return the HEAD commit SHA of the repo."""
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'git rev-parse HEAD failed (rc={rc})'
    return out.strip()


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


def _make_soft_floor_config(**overrides: Any) -> GitConfig:
    """Build a GitConfig with the θ soft floor enabled and canonical settings."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        warm_lane_soft_floor=True,
        warm_lane_min_free_gib=50,
        warm_lane_min_free_inodes=500_000,
        warm_lane_soft_free_gib=500,
        warm_lane_soft_free_inodes=5_000_000,
        **overrides,
    )


class TestWarmLaneSoftFloorConfig:
    """GitConfig soft-floor knobs: defaults + soft>hard validator (step-1)."""

    def test_defaults(self):
        config = GitConfig()
        assert config.warm_lane_soft_floor is False
        assert config.warm_lane_soft_free_gib == 500
        assert config.warm_lane_soft_free_inodes == 5_000_000

    def test_soft_floor_disabled_accepts_soft_below_hard_gib(self):
        """Validator is only enforced when warm_lane_soft_floor=True."""
        config = GitConfig(
            warm_lane_soft_floor=False,
            warm_lane_min_free_gib=50,
            warm_lane_soft_free_gib=10,
        )
        assert config.warm_lane_soft_free_gib == 10

    def test_soft_floor_disabled_accepts_soft_below_hard_inodes(self):
        config = GitConfig(
            warm_lane_soft_floor=False,
            warm_lane_min_free_inodes=500_000,
            warm_lane_soft_free_inodes=1_000,
        )
        assert config.warm_lane_soft_free_inodes == 1_000

    def test_soft_floor_enabled_equal_gib_raises(self):
        with pytest.raises(ValidationError):
            GitConfig(
                warm_lane_soft_floor=True,
                warm_lane_min_free_gib=50,
                warm_lane_soft_free_gib=50,
            )

    def test_soft_floor_enabled_below_hard_gib_raises(self):
        with pytest.raises(ValidationError):
            GitConfig(
                warm_lane_soft_floor=True,
                warm_lane_min_free_gib=50,
                warm_lane_soft_free_gib=10,
            )

    def test_soft_floor_enabled_equal_inodes_raises(self):
        with pytest.raises(ValidationError):
            GitConfig(
                warm_lane_soft_floor=True,
                warm_lane_min_free_inodes=500_000,
                warm_lane_soft_free_inodes=500_000,
            )

    def test_soft_floor_enabled_below_hard_inodes_raises(self):
        with pytest.raises(ValidationError):
            GitConfig(
                warm_lane_soft_floor=True,
                warm_lane_min_free_inodes=500_000,
                warm_lane_soft_free_inodes=1_000,
            )

    def test_soft_floor_enabled_valid_combo_accepted(self):
        config = GitConfig(
            warm_lane_soft_floor=True,
            warm_lane_min_free_gib=50,
            warm_lane_soft_free_gib=500,
            warm_lane_min_free_inodes=500_000,
            warm_lane_soft_free_inodes=5_000_000,
        )
        assert config.warm_lane_soft_floor is True


@pytest.mark.asyncio
class TestRunWarmLaneSoftGuard:
    """Unit tests for GitOps._run_warm_lane_soft_guard() (step-3)."""

    async def test_cli_contract_soft_flags_and_thresholds(self, git_repo: Path):
        """check receives --soft plus the hard AND soft threshold flags."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [3])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        rc = await git_ops._run_warm_lane_soft_guard()

        assert rc == 3
        assert _subcommands(git_repo) == ['check']
        line = _read_call_log(git_repo)[0]
        parts = line.split()

        assert '--mount' in parts
        assert parts[parts.index('--mount') + 1] == str(git_ops.worktree_base)

        assert '--min-free-gib' in parts
        assert parts[parts.index('--min-free-gib') + 1] == '50'
        assert '--min-free-inodes' in parts
        assert parts[parts.index('--min-free-inodes') + 1] == '500000'

        assert '--soft' in parts
        assert '--soft-free-gib' in parts
        assert parts[parts.index('--soft-free-gib') + 1] == '500'
        assert '--soft-free-inodes' in parts
        assert parts[parts.index('--soft-free-inodes') + 1] == '5000000'

    async def test_returns_healthy_rc(self, git_repo: Path):
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [0])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        rc = await git_ops._run_warm_lane_soft_guard()

        assert rc == 0

    async def test_returns_hard_pressure_rc(self, git_repo: Path):
        """rc=75 (hard pressure) is returned as-is — ε already owns that path."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [75])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        rc = await git_ops._run_warm_lane_soft_guard()

        assert rc == 75

    async def test_absent_script_returns_127_no_log(self, git_repo: Path):
        """No scripts committed — guard is absent (rc 127); nothing logged."""
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        rc = await git_ops._run_warm_lane_soft_guard()

        assert rc == 127
        assert _read_call_log(git_repo) == []


@pytest.mark.asyncio
class TestWarmLaneSoftPressureDefer:
    """Unit tests for GitOps._warm_lane_soft_pressure_defer() (step-5)."""

    async def test_soft_pressure_rc3_returns_true(self, git_repo: Path):
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [3])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._warm_lane_soft_pressure_defer('A')

        assert result is True, 'rc=3 (soft pressure) must defer (True)'

    @pytest.mark.parametrize('stub_rc', [0, 75, 127, 2, 9])
    async def test_non_soft_pressure_rc_returns_false(
        self, git_repo: Path, stub_rc: int,
    ):
        """Fail-open: healthy(0)/hard(75)/absent(127)/usage(2)/unknown(9) never defer."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [stub_rc])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._warm_lane_soft_pressure_defer('A')

        assert result is False, f'rc={stub_rc} must never defer (fail-open)'

    async def test_absent_script_returns_false(self, git_repo: Path):
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._warm_lane_soft_pressure_defer('A')

        assert result is False, 'Absent guard script must fail-open (False)'

    async def test_defer_emits_structured_warning_journal_line(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        """B10: the user-observable θ defer signal — a WARNING naming the
        branch, mentioning 'soft' and 'deferring'."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [3])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = await git_ops._warm_lane_soft_pressure_defer('mybranch')

        assert result is True
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'mybranch' in t and 'soft' in t.lower() and 'deferring' in t.lower()
            for t in warning_texts
        ), f'Expected a soft-floor defer WARNING naming the branch; got: {warning_texts}'


@pytest.mark.asyncio
class TestAcquireWarmLaneSoftFloor:
    """Integration: acquire_warm_lane() is gated by θ's soft-floor throttle
    (step-7), which runs AFTER the ε hard-floor disk-guard and BEFORE
    acquire_for — mirrors TestAcquireWarmLaneDiskGuard one floor earlier."""

    async def test_soft_pressure_enum_member_exists(self):
        """WarmLaneUnavailable.SOFT_PRESSURE is a distinct discriminant,
        separate from DISK_PRESSURE (θ vs ε)."""
        assert hasattr(WarmLaneUnavailable, 'SOFT_PRESSURE')
        assert WarmLaneUnavailable.SOFT_PRESSURE is not WarmLaneUnavailable.DISK_PRESSURE

    async def test_fresh_branch_soft_pressure_returns_soft_pressure(
        self, git_repo: Path,
    ):
        """Fresh (unmapped) branch + soft rc=3 ⇒ SOFT_PRESSURE; lane stays FREE."""
        from orchestrator.warm_lane_pool import LaneState

        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [3])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        result = await git_ops.acquire_warm_lane('A', start_ref)

        assert result is WarmLaneUnavailable.SOFT_PRESSURE, (
            f'Expected SOFT_PRESSURE; got {result!r}'
        )
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.state(
            git_ops.worktree_base / '_lane-0'
        ) == LaneState.FREE, 'Lane-0 must stay FREE when soft-throttled (defer, not acquire)'

    async def test_fresh_branch_soft_pressure_lane_dir_not_created(
        self, git_repo: Path,
    ):
        """SOFT_PRESSURE path: no worktree-add/seed ran — lane dir absent
        (mirrors test_still_pressured_lane_dir_not_created)."""
        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [3])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        await git_ops.acquire_warm_lane('A', start_ref)

        assert not (git_ops.worktree_base / '_lane-0').exists(), (
            'Lane dir must not be created when soft-floor admission defers'
        )

    async def test_reuse_not_throttled_even_under_soft_pressure(
        self, git_repo: Path,
    ):
        """A branch already mapped in the pool (reuse) proceeds unthrottled
        even when the soft guard would defer — only a FRESH allocation defers."""
        await _add_all_warm_lane_scripts(git_repo)
        # First acquire (fresh, unmapped): rc=0 healthy → succeeds, maps 'A'.
        # Second acquire (reuse, mapped): rc=3 pre-loaded but must NEVER be
        # consumed — assignment_for('A') is no longer None, so θ's gate
        # short-circuits before invoking the soft guard at all.
        _write_check_exits(git_repo, [0, 3])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        first = await git_ops.acquire_warm_lane('A', start_ref)
        assert isinstance(first, WorktreeInfo), f'First acquire must succeed; got {first!r}'
        assert git_ops.warm_lane_pool is not None
        assert git_ops.warm_lane_pool.assignment_for('A') is not None, (
            'Branch A must be mapped after a successful fresh acquire'
        )

        second = await git_ops.acquire_warm_lane('A', start_ref)

        assert isinstance(second, WorktreeInfo), (
            f'Reuse must NOT be soft-throttled; got {second!r}'
        )
        assert _subcommands(git_repo) == ['check'], (
            'θ soft guard must be consulted exactly once (the first, fresh '
            f'acquire) and never on reuse; got {_subcommands(git_repo)}'
        )

    async def test_knob_off_soft_guard_not_invoked(
        self, git_repo: Path,
    ):
        """warm_lane_soft_floor=False → soft guard script never runs; acquire proceeds."""
        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [3])  # Would defer if wrongly consulted
        config = GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
            warm_lane_soft_floor=False,  # master knob OFF → byte-identical
        )
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        result = await git_ops.acquire_warm_lane('A', start_ref)

        assert isinstance(result, WorktreeInfo), (
            f'Knob-off must return WorktreeInfo; got {result!r}'
        )
        assert _read_call_log(git_repo) == [], (
            'Soft guard script must NOT run when warm_lane_soft_floor=False'
        )

    async def test_hard_pressure_precedence_soft_not_consulted(
        self, git_repo: Path,
    ):
        """ε hard rc=75 (still pressured after reclaim) ⇒ DISK_PRESSURE; θ's
        soft guard is never consulted — the hard path wins/is unchanged."""
        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [75, 75])
        config = _make_soft_floor_config(warm_lane_disk_guard=True)
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)
        start_ref = await _get_head(git_repo)

        result = await git_ops.acquire_warm_lane('A', start_ref)

        assert result is WarmLaneUnavailable.DISK_PRESSURE, (
            f'ε hard floor must take precedence; got {result!r}'
        )
        check_lines = [
            line for line in _read_call_log(git_repo) if line.split()[0] == 'check'
        ]
        assert len(check_lines) == 2, (
            f'Expected exactly 2 ε check calls (check→reclaim→check); got {check_lines}'
        )
        assert all('--soft' not in line for line in check_lines), (
            f'θ soft guard must not run once ε hard-floor already blocked; got {check_lines}'
        )
