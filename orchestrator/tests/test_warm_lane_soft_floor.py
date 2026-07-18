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
from unittest.mock import patch

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


def _write_audit_stub(scripts_dir: Path) -> None:
    """Write a warm-lane-audit.sh stub into scripts_dir.

    Appends "audit <argv>" to <repo>/.test_audit_call_log, then cats
    <repo>/.test_audit_output to stdout (if present) and exits with the
    code in <repo>/.test_audit_exit (0 if absent). Mirrors
    _write_disk_guard_stubs's call-log + external-fixture-file shape, one
    script over (α, not ε/θ) — the fixture files (rather than embedding
    output in the script text) sidestep bash string-escaping for
    multi-line stdout.
    """
    audit = scripts_dir / 'warm-lane-audit.sh'
    audit.write_text(
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'ROOT="$(dirname "$DIR")"\n'
        'echo "audit $*" >> "$ROOT/.test_audit_call_log"\n'
        'if [ -f "$ROOT/.test_audit_output" ]; then\n'
        '    cat "$ROOT/.test_audit_output"\n'
        'fi\n'
        'rc=0\n'
        'if [ -f "$ROOT/.test_audit_exit" ]; then\n'
        '    rc="$(cat "$ROOT/.test_audit_exit")"\n'
        'fi\n'
        'exit "${rc:-0}"\n'
    )
    audit.chmod(0o755)


async def _add_audit_stub_script(repo: Path) -> None:
    """Commit a stub warm-lane-audit.sh into repo/scripts/."""
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    _write_audit_stub(scripts_dir)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add audit stub script'], cwd=repo)


def _set_audit_output(repo: Path, output: str, exit_code: int = 0) -> None:
    """Configure the audit stub's stdout + exit code for its next invocation(s)."""
    (repo / '.test_audit_output').write_text(output)
    (repo / '.test_audit_exit').write_text(str(exit_code))


def _read_audit_call_log(repo: Path) -> list[str]:
    """Return all non-empty lines from <repo>/.test_audit_call_log."""
    log = repo / '.test_audit_call_log'
    if not log.exists():
        return []
    return [line for line in log.read_text().splitlines() if line.strip()]


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

    async def test_unexpected_rc_logs_warning(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        """Amendment (reviewer_comprehensive test-coverage, git_ops.py:3055):
        an exit code outside {0, 3, 75} both propagates as the return value
        AND is logged as a WARNING — this script-contract-violation branch
        was previously untested (mirrors the defer WARNING test's caplog
        pattern)."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [2])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            rc = await git_ops._run_warm_lane_soft_guard()

        assert rc == 2
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            '_run_warm_lane_soft_guard' in t and '2' in t for t in warning_texts
        ), (
            'Expected an unexpected-rc WARNING from _run_warm_lane_soft_guard; '
            f'got: {warning_texts}'
        )


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

    @pytest.mark.parametrize('stub_rc', [0, 127, 2, 9])
    async def test_non_soft_pressure_rc_returns_false(
        self, git_repo: Path, stub_rc: int,
    ):
        """Fail-open: healthy(0)/absent(127)/usage(2)/unknown(9) never defer.

        rc=75 is deliberately excluded here — it is never healthy, so it
        DEFERS (unconditionally, independent of warm_lane_disk_guard); see
        TestWarmLaneSoftPressureDeferHardRc75Gap below (amendment,
        reviewer_comprehensive robustness).
        """
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
class TestWarmLaneSoftPressureDeferHardRc75Gap:
    """Amendment (reviewer_comprehensive robustness): closes a below-hard-
    floor protection gap. The soft-guard script is invoked with BOTH the hard
    and soft thresholds, so if free space/inodes are actually below the HARD
    floor it returns rc=75 (hard pressure takes precedence over soft per the
    reify contract) even though only the soft check ran. rc=75 is never
    healthy, so `_warm_lane_soft_pressure_defer` now defers on it
    UNCONDITIONALLY — independent of the warm_lane_disk_guard (ε) knob:

    - ε disabled (soft-only config): nothing else observes the hard-floor
      signal for a fresh allocation, so deferring here is the ONLY thing
      preventing an allocation into a below-hard-floor disk (the ENOSPC/
      SIGBUS condition ε exists to prevent).
    - ε enabled: ε's own check short-circuits to DISK_PRESSURE upstream on a
      genuine below-hard-floor condition, so this arm only fires in the
      narrow TOCTOU window where free space fell below the hard floor BETWEEN
      ε's check and the soft guard. Deferring there (rather than failing open
      into a fresh below-hard-floor allocation) closes that window and is
      strictly safer.

    Either way the defer is pure backpressure (WarmLaneSoftPressure REQUEUE),
    never ε's exit-75/WarmLaneDiskPressure fault path, and ε's hard path stays
    byte-identical.
    """

    async def test_rc75_defers_when_disk_guard_disabled(self, git_repo: Path):
        """Soft-only config, disk critically low (rc=75) ⇒ defers
        (backpressure), rather than silently allocating below the hard
        floor."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [75])
        config = _make_soft_floor_config(warm_lane_disk_guard=False)
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._warm_lane_soft_pressure_defer('A')

        assert result is True, (
            'rc=75 with warm_lane_disk_guard disabled must defer — nothing '
            'else backpressures a fresh allocation below the hard floor'
        )

    async def test_rc75_defers_when_disk_guard_enabled(self, git_repo: Path):
        """TOCTOU-closing case: even with ε enabled, rc=75 observed by the
        soft guard defers unconditionally. In practice ε short-circuits
        upstream on a genuine below-hard-floor condition, so this fires only
        in the narrow window where free space dropped between ε's check and
        the soft guard — deferring is strictly safer than failing open into a
        fresh below-hard-floor allocation (amendment, reviewer_comprehensive
        robustness)."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [75])
        config = _make_soft_floor_config(warm_lane_disk_guard=True)
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._warm_lane_soft_pressure_defer('A')

        assert result is True, (
            'rc=75 must defer unconditionally — it is never healthy, so '
            'deferring (backpressure) is strictly safer than failing open '
            'into a fresh allocation below the hard floor'
        )

    async def test_rc75_defer_emits_warning_naming_hard_pressure(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        """The rc=75 defer's WARNING is distinguishable from the ordinary
        soft-pressure (rc=3) one — it surfaces that the HARD floor, not the
        soft floor, was actually breached."""
        await _add_disk_guard_scripts(git_repo)
        _write_check_exits(git_repo, [75])
        config = _make_soft_floor_config(warm_lane_disk_guard=False)
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            result = await git_ops._warm_lane_soft_pressure_defer('gapbranch')

        assert result is True
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'gapbranch' in t and 'deferring' in t.lower() and 'hard' in t.lower()
            for t in warning_texts
        ), f'Expected a hard-pressure gap-closing defer WARNING; got: {warning_texts}'


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


@pytest.mark.asyncio
class TestWarmLaneSoftPressureException:
    """WarmLaneSoftPressure exception + create_worktree mapping (step-9).

    Mirrors test_warm_lane_disk_guard.py's TestWarmLaneDiskGuardE2E ---
    ``test_create_worktree_raises_warm_lane_disk_pressure`` --- one floor
    earlier. The disposition-table row assertion lives in
    test_block_disposition.py (mirroring the WarmLanePoolExhausted /
    WarmLaneDiskPressure row tests there), not here.
    """

    async def test_warm_lane_soft_pressure_is_a_warm_lane_requeue_subclass(self):
        """WarmLaneSoftPressure exists in git_ops and IS-A WarmLaneRequeue,
        so it rides the existing `except WarmLaneRequeue` requeue handler
        (workflow.py) with zero new except-clause/harness plumbing (inv.11)."""
        from orchestrator.git_ops import WarmLaneRequeue, WarmLaneSoftPressure

        assert issubclass(WarmLaneSoftPressure, WarmLaneRequeue)

    async def test_create_worktree_raises_warm_lane_soft_pressure(
        self, git_repo: Path,
    ):
        """Fresh branch + soft-floor enabled + soft rc=3 ⇒ create_worktree
        raises WarmLaneSoftPressure (mirrors
        test_create_worktree_raises_warm_lane_disk_pressure one floor
        earlier)."""
        from orchestrator.git_ops import WarmLaneSoftPressure

        await _add_all_warm_lane_scripts(git_repo)
        _write_check_exits(git_repo, [3])
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        with pytest.raises(WarmLaneSoftPressure):
            await git_ops.create_worktree('A')


@pytest.mark.asyncio
class TestRunWarmLaneAudit:
    """Unit tests for GitOps._run_warm_lane_audit() (step-11, α observability).

    α (task 2443, §9.5 inv.12) is OBSERVABILITY-ONLY: this wrapper's sole
    purpose is to surface the audit's trailing HEADROOM summary line so the
    θ defer journal line (step-14) can be enriched with pool context — it
    never gates an admission decision. Mirrors the
    _run_warm_lane_disk_guard/_run_warm_lane_soft_guard fail-soft shape
    (absent script/non-zero exit/exception -> None sentinel; never raises).
    """

    async def test_returns_headroom_line(self, git_repo: Path):
        await _add_audit_stub_script(git_repo)
        headroom = (
            'HEADROOM resident=1 assigned=0 free=1 reclaimable=0 leaked=0 '
            'leak_unknown=0 divergent_gib=2 free_gib=500 budget_gib=333'
        )
        _set_audit_output(
            git_repo,
            'lane=_lane-0 role=lane assigned=FREE branch=(detached) '
            'status=unknown recoverable=ORPHAN dirty=clean divergent_gib=2 '
            'age_min=5 classification=PRESERVED-OK\n' + headroom + '\n',
        )
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._run_warm_lane_audit()

        assert result == headroom

    async def test_absent_script_returns_none(self, git_repo: Path):
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._run_warm_lane_audit()

        assert result is None

    async def test_nonzero_exit_returns_none(self, git_repo: Path):
        """A non-zero exit degrades to None even with HEADROOM-shaped stdout."""
        await _add_audit_stub_script(git_repo)
        _set_audit_output(
            git_repo,
            'HEADROOM resident=0 assigned=0 free=0 reclaimable=0 leaked=0 '
            'leak_unknown=0 divergent_gib=0 free_gib=0 budget_gib=0\n',
            exit_code=2,
        )
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._run_warm_lane_audit()

        assert result is None

    async def test_no_headroom_line_returns_none(self, git_repo: Path):
        await _add_audit_stub_script(git_repo)
        _set_audit_output(git_repo, 'lane=_lane-0 role=lane assigned=FREE\n')
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        result = await git_ops._run_warm_lane_audit()

        assert result is None

    async def test_unexpected_exception_returns_none(self, git_repo: Path):
        """Never raises: mirrors _run_warm_lane_disk_guard's except-Exception
        fail-soft branch (test_git_ops.py's warm_lane_ref_is_degenerate
        precedent uses the same patch('orchestrator.git_ops._run', ...) shape)."""
        await _add_audit_stub_script(git_repo)
        _set_audit_output(git_repo, 'HEADROOM resident=0\n')
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        with patch('orchestrator.git_ops._run', side_effect=RuntimeError('boom')):
            result = await git_ops._run_warm_lane_audit()

        assert result is None

    async def test_read_only_invocation_no_reset_or_reclaim_subcommand(
        self, git_repo: Path,
    ):
        """Read-only contract (α/inv.12): the wrapper invokes the audit with
        no reset/reclaim subcommand or flag --- only --mount."""
        await _add_audit_stub_script(git_repo)
        _set_audit_output(git_repo, 'HEADROOM resident=0\n')
        config = _make_soft_floor_config()
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        await git_ops._run_warm_lane_audit()

        call_log = _read_audit_call_log(git_repo)
        assert len(call_log) == 1, f'Expected exactly one audit invocation; got {call_log}'
        line = call_log[0]
        assert 'reset' not in line and 'reclaim' not in line, (
            f'Audit invocation must be read-only (no reset/reclaim); got: {line!r}'
        )
        parts = line.split()
        assert '--mount' in parts
        assert parts[parts.index('--mount') + 1] == str(git_ops.worktree_base)


@pytest.mark.asyncio
class TestB10SoftFloorCapstone:
    """B10 end-to-end capstone (step-13): PRD contract §9.5 boundary test B10.

    Stubs ALL warm-lane scripts (seed + disk-guard + audit) and drives
    create_worktree() top-to-bottom for a FRESH branch with the hard floor
    healthy (ε ``check`` => 0) but the soft floor pressured (θ
    ``check --soft`` => 3): no new divergent lane is allocated, the hard
    floor is never reached, and the θ defer WARNING journal line is
    enriched with the α HEADROOM detail.
    """

    async def test_b10_soft_pressure_defers_without_reaching_hard_floor(
        self, git_repo: Path, caplog: pytest.LogCaptureFixture,
    ):
        from orchestrator.git_ops import WarmLaneSoftPressure
        from orchestrator.warm_lane_pool import LaneState
        from orchestrator.workflow_types import RequeueKind, classify_failure

        await _add_all_warm_lane_scripts(git_repo)
        await _add_audit_stub_script(git_repo)
        headroom = (
            'HEADROOM resident=1 assigned=1 free=0 reclaimable=0 leaked=0 '
            'leak_unknown=0 divergent_gib=3 free_gib=777 budget_gib=518'
        )
        _set_audit_output(
            git_repo,
            'lane=_lane-0 role=lane assigned=ASSIGNED branch=task/NEW '
            'status=non-terminal recoverable=ORPHAN dirty=clean '
            'divergent_gib=3 age_min=1 classification=LIVE\n' + headroom + '\n',
        )
        # First 'check' (ε hard floor, no --soft) => 0 (healthy, never
        # reaches exit-75/reclaim). Second 'check --soft' (θ soft floor)
        # => 3 (soft pressure, between the floors).
        _write_check_exits(git_repo, [0, 3])
        config = _make_soft_floor_config(warm_lane_disk_guard=True)
        git_ops = GitOps(config, git_repo, warm_lane_pool_size=1)

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'),
            pytest.raises(WarmLaneSoftPressure) as exc_info,
        ):
            await git_ops.create_worktree('NEW')

        # (b) no new divergent lane allocated: pool lane stays FREE, no lane dir.
        assert git_ops.warm_lane_pool is not None
        lane_dir = git_ops.worktree_base / '_lane-0'
        assert git_ops.warm_lane_pool.state(lane_dir) == LaneState.FREE, (
            'Lane-0 must stay FREE — soft-floor defer must not acquire a lane'
        )
        assert not lane_dir.exists(), (
            'No lane dir may be created on the soft-floor defer path'
        )

        # (c) hard floor never reached: exactly one ε check (rc=0, healthy,
        # no reclaim) and one θ check --soft (rc=3); no exit-75/reclaim path.
        check_lines = [
            line for line in _read_call_log(git_repo) if line.split()[0] == 'check'
        ]
        reclaim_lines = [
            line for line in _read_call_log(git_repo) if line.split()[0] == 'reclaim'
        ]
        assert len(check_lines) == 2, f'Expected 2 check invocations; got {check_lines}'
        assert '--soft' not in check_lines[0], (
            f'First check must be the ε hard-floor check (no --soft); got {check_lines[0]!r}'
        )
        assert '--soft' in check_lines[1], (
            f'Second check must be the θ soft-floor check; got {check_lines[1]!r}'
        )
        assert reclaim_lines == [], (
            f'Hard floor was healthy (rc=0) — GC reclaim must never run; got {reclaim_lines}'
        )

        # (d) the θ defer WARNING journal line names the branch, mentions
        # soft/deferring, AND is enriched verbatim with the α HEADROOM line.
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            'NEW' in t and 'soft' in t.lower() and 'deferring' in t.lower()
            and headroom in t
            for t in warning_texts
        ), f'Expected an α-enriched θ defer WARNING; got: {warning_texts}'

        # disposition -> outcome contract: REQUEUE (backpressure), not exit-75,
        # and never counts against the requeue cap (inv.11).
        disp = classify_failure(exc_info.value)
        assert disp.requeue_kind is RequeueKind.REQUEUE
        assert disp.counts_against_requeue_cap is False
