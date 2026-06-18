"""Tests for reify PRD boundary-test B9 — merge-spec warm-lane slot correctness.

Task η (1789): K>1 speculative LOCAL verifies run WARM in parallel on one box,
each in a distinct `_spec-` CoW-seeded lane.  `main` advances strictly
serial+ordered via CAS.  A cold from-scratch safety-valve verify agrees — else
a HARD ALARM (born-at-L2).

B9 capstone: speculation_depth=2 + merge_spec_warm_lane_pool on →
  * each verify WARM in a DISTINCT `_spec-` lane (per-lane seed invoked, lanes
    ASSIGNED concurrently — mechanism, not wall-time)
  * `main` advances strictly serial+ordered via CAS
  * cold safety-valve agrees → no alarm; injected divergence → born-at-L2 alarm

All wall-time assertions are improvement-direction/recorded-delta only (PRD §G6):
tests assert MECHANISM (seed invoked, lanes distinct+ASSIGNED, advance ordering/
CAS, alarm fired on injected divergence), never timing.

Shared helpers
--------------
_write_recording_script(lane, name)
    Drop an executable ``<lane>/scripts/<name>`` that appends its argv (space-
    joined) to ``<lane>/scripts/<name>.argv``.  Mirrors the fake seed-warm-lane.sh
    pattern from test_warm_lane_pool.py::TestSeedWarmLane.

spec_git_repo (fixture)
    Real git repo initialised at ``tmp_path/repo`` with worktrees base
    ``repo/.worktrees``.  Used by GitOps + WarmLanePool tests.

_make_spec_git_config(*, on, tmp_path) (helper)
    Build a GitConfig with merge_spec_warm_lane_pool=on.  Wraps GitConfig() with
    the common fields set so each test does not repeat boilerplate.

_make_escalation_queue(*, has_open) (helper)
    MagicMock escalation_queue with the standard API stubs (make_id, has_open_l1,
    get_by_task, submit).  Mirrors test_merge_queue_warm_cold_shadow.py.

_patch_cold_shadow_verify(monkeypatch, return_value) (helper)
    Patch ``_run_cold_shadow_verify`` to return a controlled per-test dict so
    cold-shadow assertions do not require a real git repo or runner subprocess.
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future, pydantic_spec  # noqa: F401

from orchestrator.config import GitConfig, OrchestratorConfig  # noqa: F401
from orchestrator.git_ops import GitOps, WorktreeInfo, _run  # noqa: F401
from orchestrator.merge_queue import (  # noqa: F401
    SpeculativeMergeWorker,
    _maybe_schedule_shadow_compare,
    _submit_shadow_divergence_escalation,
)
from orchestrator.warm_lane_pool import LaneState, WarmLanePool  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _write_recording_script(lane: Path, name: str) -> Path:
    """Install an executable ``<lane>/scripts/<name>`` that records its argv.

    Each invocation appends the space-joined argv to
    ``<lane>/scripts/<name>.argv``, one line per call.  The script exits 0.

    Returns the Path of the created script so callers can inspect the argv
    file at ``script.parent / (script.name + '.argv')``.

    Mirrors the fake seed-warm-lane.sh pattern in
    ``test_warm_lane_pool.py::TestSeedWarmLane``.

    Usage::

        script = _write_recording_script(lane, 'seed-warm-lane.sh')
        argv_file = script.parent / (script.name + '.argv')
        ...
        recorded = argv_file.read_text().strip()
        assert recorded == f'{base_target} {lane} --reset-in-place'
    """
    scripts_dir = lane / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / name
    argv_file = scripts_dir / (name + '.argv')
    script.write_text(
        f'#!/usr/bin/env bash\necho "$@" >> {argv_file}\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


# ---------------------------------------------------------------------------
# Git repo fixture
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    """Initialise a bare-minimum git repository with one commit on ``main``."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def spec_git_repo(tmp_path: Path) -> Path:
    """Real git repository at ``tmp_path/repo`` ready for worktree operations.

    Returns the repo Path.  The worktree base (``repo/.worktrees``) is created
    lazily by git worktree operations or by GitOps construction.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


# ---------------------------------------------------------------------------
# GitConfig helper
# ---------------------------------------------------------------------------


def _make_spec_git_config(*, on: bool = True, **extra) -> GitConfig:
    """Build a GitConfig with ``merge_spec_warm_lane_pool=on``.

    Passes ``**extra`` through to GitConfig() so individual tests can override
    fields (e.g. ``warm_verify_shadow_compare=True``) without repeating all
    the boilerplate fields.
    """
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        merge_spec_warm_lane_pool=on,
        **extra,
    )


# ---------------------------------------------------------------------------
# Escalation-queue spy
# ---------------------------------------------------------------------------


def _make_escalation_queue(*, has_open: bool = False) -> MagicMock:
    """Return a MagicMock escalation_queue with standard API stubs.

    Mirrors ``_make_escalation_queue`` in
    ``test_merge_queue_warm_cold_shadow.py``.
    """
    q = MagicMock()
    q.make_id = MagicMock(return_value='esc-spec-shadow-1')
    q.has_open_l1 = MagicMock(return_value=has_open)
    q.get_by_task = MagicMock(return_value=None)
    q.submit = MagicMock()
    return q


# ---------------------------------------------------------------------------
# Cold-shadow-verify patcher
# ---------------------------------------------------------------------------


def _patch_cold_shadow_verify(monkeypatch, return_value: dict[str, str]) -> None:
    """Patch ``_run_cold_shadow_verify`` to return a controlled per-test dict.

    Allows shadow-valve tests to inject warm==cold (parity) or warm!=cold
    (divergence) without a real git repo or runner subprocess.

    Usage::

        _patch_cold_shadow_verify(monkeypatch, {'test_a': 'pass', 'test_b': 'pass'})
    """
    mock = AsyncMock(return_value=return_value)
    monkeypatch.setattr(
        'orchestrator.merge_queue._run_cold_shadow_verify',
        mock,
    )


# ===========================================================================
# Step-7: RED — GitOps.acquire_spec_lane (create-once path, seed invoked)
# ===========================================================================


async def _add_recording_seed_to_repo(repo: Path) -> None:
    """Commit a recording seed-warm-lane.sh into the repo at HEAD.

    The script appends its argv (space-joined) to
    ``<lane_dir>/scripts/seed-warm-lane.sh.argv`` so tests can assert the
    argv without filesystem side-effects on the base directory.

    Mirrors the fake-script pattern from TestSeedWarmLane in
    test_warm_lane_pool.py, but committed to the repo so the script is
    available inside the worktree that acquire_spec_lane creates on the
    create-once path.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'seed-warm-lane.sh'
    script.write_text(
        '#!/usr/bin/env bash\n'
        '# argv: <base_target> <lane_dir> <mode>\n'
        'ARGV_FILE="$2/scripts/seed-warm-lane.sh.argv"\n'
        'echo "$@" >> "$ARGV_FILE"\n'
    )
    script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add recording seed-warm-lane.sh'], cwd=repo)


@pytest.mark.asyncio
class TestAcquireSpecLane:
    """GitOps.acquire_spec_lane: create-once path seeds lane with '--reset-in-place'."""

    async def test_acquire_spec_lane_returns_warm_tuple(self, spec_git_repo: Path):
        """acquire_spec_lane returns (lane_path, warm=True) for the assigned _spec- lane."""
        await _add_recording_seed_to_repo(spec_git_repo)
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=2)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        result = await git_ops.acquire_spec_lane(merge_commit)

        lane_path, warm = result
        assert warm is True, f'Expected warm=True, got {warm!r}'
        expected_lane = git_ops.worktree_base / '_spec-0'
        assert lane_path == expected_lane, (
            f'Expected lane {expected_lane}, got {lane_path!r}'
        )

    async def test_acquire_spec_lane_marks_lane_assigned(self, spec_git_repo: Path):
        """After acquire_spec_lane, the _spec-0 lane is ASSIGNED in spec_warm_lane_pool."""
        await _add_recording_seed_to_repo(spec_git_repo)
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=2)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        lane_path, _ = await git_ops.acquire_spec_lane(merge_commit)

        assert git_ops.spec_warm_lane_pool is not None
        assert git_ops.spec_warm_lane_pool._lanes[lane_path] == LaneState.ASSIGNED

    async def test_acquire_spec_lane_invokes_seed_reset_in_place(
        self, spec_git_repo: Path,
    ):
        """acquire_spec_lane invokes seed-warm-lane.sh with base_target + '--reset-in-place'."""
        await _add_recording_seed_to_repo(spec_git_repo)
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=1)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        lane_path, warm = await git_ops.acquire_spec_lane(merge_commit)

        assert warm is True
        argv_file = lane_path / 'scripts' / 'seed-warm-lane.sh.argv'
        assert argv_file.exists(), (
            f'seed-warm-lane.sh was not invoked (argv file missing at {argv_file})'
        )
        recorded = argv_file.read_text().strip()
        base_target = str(git_ops.warm_lane_base_target_path)
        expected = f'{base_target} {lane_path} --reset-in-place'
        assert recorded == expected, (
            f'Wrong seed argv: got {recorded!r}, expected {expected!r}'
        )

    async def test_acquire_spec_lane_creates_registered_worktree(
        self, spec_git_repo: Path,
    ):
        """The _spec- lane is a registered git worktree after acquire_spec_lane."""
        await _add_recording_seed_to_repo(spec_git_repo)
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=1)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        lane_path, _ = await git_ops.acquire_spec_lane(merge_commit)

        assert await git_ops._is_registered_worktree(lane_path), (
            f'Lane {lane_path} is not a registered git worktree'
        )


# ===========================================================================
# Step-9: RED — acquire_spec_lane cold fallback (pool exhausted + seed fail)
# ===========================================================================


@pytest.mark.asyncio
class TestAcquireSpecLaneFallback:
    """acquire_spec_lane falls back to cold ephemeral worktree on exhaustion or seed fail."""

    async def test_pool_exhausted_returns_cold_false(self, spec_git_repo: Path):
        """When all _spec- lanes are ASSIGNED, returns (cold_wt, False) — never None."""
        await _add_recording_seed_to_repo(spec_git_repo)
        cfg = _make_spec_git_config(on=True)
        # Pool size=1 so acquiring once exhausts it
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=1)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        # First acquire: consumes the only lane
        lane1, warm1 = await git_ops.acquire_spec_lane(merge_commit)
        assert warm1 is True, 'First acquire should be warm'

        # Second acquire: pool exhausted → cold fallback
        lane2, warm2 = await git_ops.acquire_spec_lane(merge_commit)

        assert warm2 is False, f'Expected warm=False on exhaustion, got {warm2!r}'
        assert lane2 is not None, 'acquire_spec_lane must never return None'
        # The cold fallback must be a DIFFERENT path from the warm lane
        assert lane2 != lane1, (
            f'Cold fallback path {lane2} must differ from warm lane {lane1}'
        )
        # Cold fallback is a registered worktree (ephemeral _merge-<uuid>)
        assert await git_ops._is_registered_worktree(lane2), (
            f'Cold fallback {lane2} is not a registered git worktree'
        )
        # The warm lane is still ASSIGNED (not released)
        assert git_ops.spec_warm_lane_pool is not None
        assert git_ops.spec_warm_lane_pool._lanes[lane1] == LaneState.ASSIGNED

    async def test_seed_failure_returns_cold_false(self, spec_git_repo: Path):
        """When seed fails (no seed script), falls back to cold — lane released to FREE."""
        # Do NOT add seed script — so _seed_warm_lane returns False
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=1)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        lane_path, warm = await git_ops.acquire_spec_lane(merge_commit)

        assert warm is False, f'Expected warm=False on seed failure, got {warm!r}'
        assert lane_path is not None, 'acquire_spec_lane must never return None'

    async def test_seed_failure_releases_lane_back_to_free(self, spec_git_repo: Path):
        """On seed failure the partially-acquired _spec- lane is released back to FREE."""
        # No seed script → _seed_warm_lane returns False → lane must be released
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=1)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        await git_ops.acquire_spec_lane(merge_commit)

        # After seed-failure fallback the pool's only lane should be FREE again
        assert git_ops.spec_warm_lane_pool is not None
        lane0 = git_ops.worktree_base / '_spec-0'
        assert git_ops.spec_warm_lane_pool._lanes[lane0] == LaneState.FREE, (
            'Lane must be released back to FREE after seed failure'
        )

    async def test_exhausted_never_raises(self, spec_git_repo: Path):
        """Pool exhaustion never raises — returns a usable cold worktree."""
        await _add_recording_seed_to_repo(spec_git_repo)
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=1)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        # Exhaust the pool
        await git_ops.acquire_spec_lane(merge_commit)

        # Must not raise on exhaustion
        try:
            result = await git_ops.acquire_spec_lane(merge_commit)
        except Exception as exc:
            pytest.fail(
                f'acquire_spec_lane raised {type(exc).__name__} on exhaustion: {exc}'
            )
        assert result is not None
