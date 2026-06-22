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
import contextlib
import stat
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future, pydantic_spec  # noqa: F401

from orchestrator.config import GitConfig, OrchestratorConfig  # noqa: F401
from orchestrator.event_store import EventStore, EventType  # noqa: F401
from orchestrator.git_ops import GitOps, WorktreeInfo, _run  # noqa: F401
from orchestrator.merge_queue import (  # noqa: F401
    InflightEntry,
    InflightVerifyResult,
    MergeOutcome,
    MergeRequest,
    SpeculativeItem,
    SpeculativeMergeWorker,
    _acquire_warm_verify_worktree,
    _maybe_run_drift_check,
    _maybe_schedule_shadow_compare,
    _run_cold_shadow_verify,
    _submit_shadow_divergence_escalation,
)
from orchestrator.verify import VerifyResult  # noqa: F401
from orchestrator.warm_lane_pool import LaneState, WarmLanePool  # noqa: F401
# Late-arrival integration helpers (task 1862): reuse module-level helpers
# from test_merge_queue_concurrent_verify.  These are private (_-prefixed)
# module-level functions so pytest does not collect them as tests.
from test_merge_queue_concurrent_verify import (  # noqa: F401
    _gated_runner,
    _inject_two_host_allocator,
    _make_branch_with_file,
    _make_request,
)

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


# ===========================================================================
# Step-11: RED — GitOps.release_spec_lane
# ===========================================================================


@pytest.mark.asyncio
class TestReleaseSpecLane:
    """release_spec_lane: warm lane → FREE (target retained); cold → cleanup; idempotent."""

    async def test_release_warm_lane_flips_to_free(self, spec_git_repo: Path):
        """Releasing a warm _spec- lane sets its state back to FREE in spec_warm_lane_pool."""
        await _add_recording_seed_to_repo(spec_git_repo)
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=1)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        lane_path, warm = await git_ops.acquire_spec_lane(merge_commit)
        assert warm is True

        # Lane is ASSIGNED — release it
        await git_ops.release_spec_lane(lane_path, warm=warm)

        assert git_ops.spec_warm_lane_pool is not None
        assert git_ops.spec_warm_lane_pool._lanes[lane_path] == LaneState.FREE

    async def test_release_warm_lane_retains_worktree(self, spec_git_repo: Path):
        """release_spec_lane does NOT remove the worktree (target/ warmth retained)."""
        await _add_recording_seed_to_repo(spec_git_repo)
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=1)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        lane_path, warm = await git_ops.acquire_spec_lane(merge_commit)

        await git_ops.release_spec_lane(lane_path, warm=warm)

        # Worktree must still exist on disk
        assert lane_path.exists(), (
            f'release_spec_lane deleted the warm lane directory {lane_path}'
        )
        # And still registered as a git worktree
        assert await git_ops._is_registered_worktree(lane_path), (
            f'release_spec_lane un-registered the warm lane {lane_path}'
        )

    async def test_release_cold_fallback_removes_worktree(self, spec_git_repo: Path):
        """Releasing a cold-fallback (warm=False) worktree removes it via cleanup."""
        # No seed script → seed fails → cold fallback
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=1)

        _, merge_commit, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=spec_git_repo,
        )
        merge_commit = merge_commit.strip()

        cold_path, warm = await git_ops.acquire_spec_lane(merge_commit)
        assert warm is False

        assert cold_path.exists(), 'cold fallback path should exist before release'

        await git_ops.release_spec_lane(cold_path, warm=warm)

        assert not cold_path.exists(), (
            f'release_spec_lane(warm=False) must remove the cold worktree {cold_path}'
        )

    async def test_release_idempotent_on_free_lane(self, spec_git_repo: Path):
        """Releasing an already-FREE or unknown lane never raises (idempotent)."""
        await _add_recording_seed_to_repo(spec_git_repo)
        cfg = _make_spec_git_config(on=True)
        git_ops = GitOps(cfg, spec_git_repo, merge_spec_warm_lane_pool_size=1)

        lane_path = git_ops.worktree_base / '_spec-0'

        # Lane was never acquired — release should be a no-op
        try:
            await git_ops.release_spec_lane(lane_path, warm=True)
        except Exception as exc:
            pytest.fail(
                f'release_spec_lane raised {type(exc).__name__} on FREE lane: {exc}'
            )


# ===========================================================================
# Step-13: RED — _acquire_warm_verify_worktree speculative routing
# ===========================================================================


def _make_spec_req(tmp_path: Path, *, spec_knob: bool, persistent: bool = False) -> MagicMock:
    """Build a minimal MergeRequest stub for spec-routing tests."""
    req = MagicMock()
    req.task_id = 'task-spec-1'
    req.config = OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            persistent_merge_worktree=persistent,
            merge_spec_warm_lane_pool=spec_knob,
        ),
    )
    return req


def _make_spec_git_ops_spy(
    *,
    lane_path: Path | None = None,
    warm_path: Path | None = None,
) -> MagicMock:
    """Build a GitOps MagicMock with async spies for routing assertions."""
    git_ops = MagicMock()
    _lane = lane_path or Path('/fake/_spec-0')
    _warm = warm_path or Path('/fake/_merge-verify')
    git_ops.acquire_spec_lane = AsyncMock(return_value=(_lane, True))
    git_ops.reset_persistent_merge_worktree = AsyncMock(return_value=_warm)
    git_ops.release_spec_lane = AsyncMock()
    git_ops.cleanup_merge_worktree = AsyncMock()
    return git_ops


@pytest.mark.asyncio
class TestAcquireWarmVerifyWorktreeSpecRouting:
    """_acquire_warm_verify_worktree routes speculative LOCAL items to acquire_spec_lane."""

    async def test_speculative_routes_to_acquire_spec_lane(self, tmp_path: Path):
        """speculative=True + merge_spec_warm_lane_pool → acquire_spec_lane called."""
        req = _make_spec_req(tmp_path, spec_knob=True, persistent=True)
        git_ops = _make_spec_git_ops_spy()
        merge_wt = tmp_path / '_merge-abc'

        await _acquire_warm_verify_worktree(
            git_ops, req, merge_wt, 'deadbeef' * 5,
            safety_valve_due=False, speculative=True,
        )

        git_ops.acquire_spec_lane.assert_called_once()
        git_ops.reset_persistent_merge_worktree.assert_not_called()

    async def test_non_speculative_routes_to_reset_persistent(self, tmp_path: Path):
        """speculative=False → reset_persistent_merge_worktree (serial head), not acquire_spec_lane."""
        req = _make_spec_req(tmp_path, spec_knob=True, persistent=True)
        git_ops = _make_spec_git_ops_spy()
        merge_wt = tmp_path / '_merge-abc'

        await _acquire_warm_verify_worktree(
            git_ops, req, merge_wt, 'deadbeef' * 5,
            safety_valve_due=False, speculative=False,
        )

        git_ops.reset_persistent_merge_worktree.assert_called_once()
        git_ops.acquire_spec_lane.assert_not_called()

    async def test_spec_knob_off_ignores_speculative_flag(self, tmp_path: Path):
        """merge_spec_warm_lane_pool=False → spec branch not taken even with speculative=True."""
        req = _make_spec_req(tmp_path, spec_knob=False, persistent=False)
        git_ops = _make_spec_git_ops_spy()
        merge_wt = tmp_path / '_merge-abc'

        await _acquire_warm_verify_worktree(
            git_ops, req, merge_wt, 'deadbeef' * 5,
            safety_valve_due=False, speculative=True,
        )

        git_ops.acquire_spec_lane.assert_not_called()

    async def test_safety_valve_due_bypasses_spec_pool(self, tmp_path: Path):
        """safety_valve_due=True → spec pool bypassed even with speculative=True + knob on."""
        req = _make_spec_req(tmp_path, spec_knob=True, persistent=True)
        git_ops = _make_spec_git_ops_spy()
        merge_wt = tmp_path / '_merge-abc'

        await _acquire_warm_verify_worktree(
            git_ops, req, merge_wt, 'deadbeef' * 5,
            safety_valve_due=True, speculative=True,
        )

        git_ops.acquire_spec_lane.assert_not_called()

    async def test_speculative_warm_result_threads_back(self, tmp_path: Path):
        """Spec acquire returns (lane_path, warm=True); function threads warm signal to caller."""
        req = _make_spec_req(tmp_path, spec_knob=True, persistent=True)
        lane_path = Path('/fake/_spec-0')
        git_ops = _make_spec_git_ops_spy(lane_path=lane_path)
        merge_wt = tmp_path / '_merge-abc'

        result = await _acquire_warm_verify_worktree(
            git_ops, req, merge_wt, 'deadbeef' * 5,
            safety_valve_due=False, speculative=True,
        )

        # Result must be (path, warm) so the caller can route release correctly:
        # warm=True → release_spec_lane; warm=False → cleanup_merge_worktree.
        path, warm = result
        assert path == lane_path
        assert warm is True


# ===========================================================================
# Step-15: RED — spec-lane warm verify feeds shadow safety valve (parity case)
# ===========================================================================


def _make_minimal_worker(tmp_path: Path) -> SpeculativeMergeWorker:
    """Build a SpeculativeMergeWorker with a mock git_ops for _run_inflight_verify tests.

    Sets ``project_root`` on the mock so ``_shadow_state_path`` resolves to a
    real path under ``tmp_path`` (enabling ``_maybe_schedule_shadow_compare``
    cadence state persistence).  All git operations are left as MagicMock so
    no real worktree I/O occurs in the test.
    """
    mock_git_ops = MagicMock()
    mock_git_ops.project_root = tmp_path
    return SpeculativeMergeWorker(mock_git_ops, asyncio.Queue())


def _make_spec_item(
    tmp_path: Path,
    cfg: OrchestratorConfig,
    *,
    speculative: bool = True,
) -> SpeculativeItem:
    """Build a minimal SpeculativeItem for _run_inflight_verify tests.

    Uses a real asyncio.Future for ``req.result`` so ``_request_abandoned``
    (which checks ``.cancelled()``) returns False — preventing the abort path.
    The merge_wt is a real directory so path comparison works correctly.
    """
    result_fut: asyncio.Future = asyncio.get_running_loop().create_future()
    fake_req = MagicMock()
    fake_req.task_id = 'task-spec-shadow-15'
    fake_req.config = cfg
    fake_req.result = result_fut

    merge_wt = tmp_path / '_merge-abc'
    merge_wt.mkdir(parents=True, exist_ok=True)

    return SpeculativeItem(
        request=fake_req,
        merge_result=MagicMock(merge_commit='deadbeef01234567890a'),
        merge_wt=merge_wt,
        base_sha='aabbccdd00000000aaaa',
        speculative=speculative,
        skip_verify=False,
    )


@pytest.mark.asyncio
class TestSpecLaneWarmPathShadowParity:
    """Spec-lane warm verify must feed _maybe_schedule_shadow_compare (parity case).

    The existing shadow safety valve (κ invariant 6) is already wired in the
    speculative finalize path (merge_queue.py:8137) and fires when
    ``warm_results`` is non-empty.  For this to cover spec-lane verifies,
    ``_is_warm_path`` must be True when ``_spec_warm=True``
    (task η step-16 extends the predicate).

    Until step-16 (GREEN): ``_is_warm_path = persistent_merge_worktree and not _due``
    is False for the spec-pool-only config (``persistent_merge_worktree=False``),
    so ``on_result=None`` → ``_warm_capture`` empty → ``warm_results={}``
    → shadow no-ops → these tests FAIL (RED).
    """

    async def test_spec_lane_warm_results_captured(self, tmp_path: Path):
        """warm_results in InflightVerifyResult must be non-empty for spec-lane warm verify.

        RED: _is_warm_path=False → on_result=None → _warm_capture empty → warm_results={}.
        GREEN (step-16): _is_warm_path extended to include spec lane → populated.
        """
        # Config: spec pool on, shadow on, NO persistent worktree for serial head.
        # _is_warm_path must be True from the spec-lane path, not persistent_merge_worktree.
        cfg = OrchestratorConfig(
            project_root=tmp_path,
            git=_make_spec_git_config(
                on=True,
                warm_verify_shadow_compare=True,
                warm_verify_shadow_compare_every_n_merges=1,
            ),
        )
        worker = _make_minimal_worker(tmp_path)
        item = _make_spec_item(tmp_path, cfg, speculative=True)

        # Fake VerifyResult with parseable per-test output
        fake_vr = VerifyResult(
            passed=True,
            test_output='        PASS [0.05s] reify-spec test_warm_shadow\n',
            lint_output='',
            type_output='',
            summary='',
        )

        # Mock verify: invoke on_result callback if provided (simulates warm capture).
        # When _is_warm_path=True, on_result=_warm_capture.append is passed.
        # When _is_warm_path=False, on_result=None → callback never invoked.
        async def _mock_verify(*args, **kwargs):
            on_result = kwargs.get('on_result')
            if on_result is not None:
                on_result(fake_vr)
            return None  # verify passed

        # Fake spec lane (spec pool is on, acquire returns this path as warm)
        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)

        with patch(
            'orchestrator.merge_queue._acquire_warm_verify_worktree',
            new=AsyncMock(return_value=(fake_lane, True)),
        ), patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=_mock_verify,
        ):
            vr = await worker._run_inflight_verify(item, _make_local_lease())

        # RED: warm_results={} because _is_warm_path=False ignores _spec_warm
        # GREEN (step-16): warm_results={'spec::test_warm_shadow': 'pass'}
        assert vr.warm_results, (
            'spec-lane warm verify must populate warm_results so the shadow '
            'safety valve can compare warm vs cold per-test; '
            f'got warm_results={vr.warm_results!r}'
        )

    async def test_spec_lane_parity_shadow_fires_no_escalation(self, tmp_path: Path):
        """Shadow fires for spec-lane warm verify; warm==cold → no escalation.

        RED: warm_results={} → _maybe_schedule_shadow_compare no-ops →
        cold verify NOT called → assertion on cold_verify_calls fails.
        GREEN (step-16): warm_results populated → shadow fires → cold called →
        parity → escalation_queue.submit not called.
        """
        cfg = OrchestratorConfig(
            project_root=tmp_path,
            git=_make_spec_git_config(
                on=True,
                warm_verify_shadow_compare=True,
                warm_verify_shadow_compare_every_n_merges=1,  # always due
            ),
        )
        worker = _make_minimal_worker(tmp_path)
        item = _make_spec_item(tmp_path, cfg, speculative=True)

        fake_vr = VerifyResult(
            passed=True,
            test_output='        PASS [0.05s] reify-spec test_warm_shadow\n',
            lint_output='',
            type_output='',
            summary='',
        )

        async def _mock_verify(*args, **kwargs):
            on_result = kwargs.get('on_result')
            if on_result is not None:
                on_result(fake_vr)
            return None

        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)

        cold_verify_calls: list[tuple] = []

        # Cold verify returns SAME results as warm (parity case)
        async def _mock_cold_shadow_verify(*args, **kwargs):
            cold_verify_calls.append(args)
            return {'reify-spec test_warm_shadow': 'pass'}  # parity with warm

        escalation_queue = _make_escalation_queue()

        with patch(
            'orchestrator.merge_queue._acquire_warm_verify_worktree',
            new=AsyncMock(return_value=(fake_lane, True)),
        ), patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=_mock_verify,
        ), patch(
            'orchestrator.merge_queue._run_cold_shadow_verify',
            new=_mock_cold_shadow_verify,
        ):
            vr = await worker._run_inflight_verify(item, _make_local_lease())

            # Drive _maybe_schedule_shadow_compare with the captured warm_results
            await _maybe_schedule_shadow_compare(
                worker,
                worker._git_ops,
                item.request,
                'deadbeef01234567890a',
                warm_results=vr.warm_results,
                escalation_queue=escalation_queue,
                event_store=None,
            )
            # Drain the background shadow task (if one was spawned)
            if worker._shadow_compare_tasks:
                await asyncio.gather(*list(worker._shadow_compare_tasks))

        # RED: cold_verify_calls=[] (shadow never ran; warm_results empty)
        # GREEN (step-16): cold_verify_calls non-empty (shadow ran on spec-lane warm result)
        assert cold_verify_calls, (
            'shadow safety valve must run cold verify for spec-lane warm verify; '
            'spec-lane not treated as warm path → warm_results={} → shadow no-ops'
        )
        # Parity: no escalation submitted
        escalation_queue.submit.assert_not_called()


def _make_local_lease() -> MagicMock:
    """Build a mock HostLease with is_local=True for _run_inflight_verify tests."""
    lease = MagicMock()
    lease.is_local = True
    return lease


# ===========================================================================
# Step-17: RED — per-spec-lane DIVERGENCE → HARD ALARM; unparseable → alarm
# ===========================================================================


@pytest.mark.asyncio
class TestSpecLaneDivergenceAlarm:
    """Per-spec-lane divergence → born-at-L2 alarm; unparseable → fail-closed alarm.

    Reuses _submit_shadow_divergence_escalation semantics (severity=critical,
    orchestrator-prefixed agent_role, category=risk_identified).

    RED conditions (pre-step-16 / pre-step-18 context):
    - Divergence: warm_results empty → shadow no-ops → no alarm.
    - Unparseable: _alarm_warm_shadow_unparseable not called (no warm capture).
    GREEN after steps-16 + 18: both paths produce an alarm.
    """

    async def test_divergence_fires_born_at_l2_alarm(self, tmp_path: Path):
        """Injected warm-pass/cold-fail divergence on a spec lane → escalation.submit."""
        cfg = OrchestratorConfig(
            project_root=tmp_path,
            git=_make_spec_git_config(
                on=True,
                warm_verify_shadow_compare=True,
                warm_verify_shadow_compare_every_n_merges=1,
            ),
        )
        escalation_spy = _make_escalation_queue()
        worker = _make_minimal_worker(tmp_path)

        item = _make_spec_item(tmp_path, cfg, speculative=True)

        # Warm: test passes
        fake_vr = VerifyResult(
            passed=True,
            test_output='        PASS [0.05s] reify-spec test_diverge\n',
            lint_output='', type_output='', summary='',
        )

        async def _mock_verify(*args, **kwargs):
            on_result = kwargs.get('on_result')
            if on_result is not None:
                on_result(fake_vr)
            return None

        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)

        # Cold: test FAILS on both first run AND re-confirmation (Option B)
        cold_call_count = 0

        async def _divergent_cold(*args, **kwargs):
            nonlocal cold_call_count
            cold_call_count += 1
            return {'reify-spec test_diverge': 'fail'}  # diverges from warm pass

        with patch(
            'orchestrator.merge_queue._acquire_warm_verify_worktree',
            new=AsyncMock(return_value=(fake_lane, True)),
        ), patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=_mock_verify,
        ), patch(
            'orchestrator.merge_queue._run_cold_shadow_verify',
            new=_divergent_cold,
        ):
            vr = await worker._run_inflight_verify(item, _make_local_lease())

            await _maybe_schedule_shadow_compare(
                worker,
                worker._git_ops,
                item.request,
                'deadbeef01234567890a',
                warm_results=vr.warm_results,
                escalation_queue=escalation_spy,
                event_store=None,
            )
            # Drain shadow tasks (re-confirmation cold run is also backgrounded)
            if worker._shadow_compare_tasks:
                await asyncio.gather(*list(worker._shadow_compare_tasks))

        # Option B re-confirmation requires two cold runs
        assert cold_call_count == 2, (
            f'Expected 2 cold runs (first + re-confirmation); got {cold_call_count}'
        )
        # Born-at-L2 alarm must be submitted for persistent divergence
        escalation_spy.submit.assert_called_once()
        esc = escalation_spy.submit.call_args[0][0]
        assert esc.severity == 'critical', (
            f'Divergence escalation must be critical; got {esc.severity!r}'
        )
        assert esc.agent_role.startswith('orchestrator-'), (
            f'Born-at-L2 requires orchestrator- prefix; got {esc.agent_role!r}'
        )
        assert esc.category == 'risk_identified', (
            f'Expected risk_identified; got {esc.category!r}'
        )

    async def test_unparseable_warm_verify_alarms(self, tmp_path: Path):
        """Unparseable warm output on a spec lane → _alarm_warm_shadow_unparseable fires.

        The fail-closed guard must never allow a silent pass (κ invariant 6).
        """
        cfg = OrchestratorConfig(
            project_root=tmp_path,
            git=_make_spec_git_config(
                on=True,
                warm_verify_shadow_compare=True,
            ),
        )
        # Worker WITH escalation_queue spy so _alarm_warm_shadow_unparseable submits
        escalation_spy = _make_escalation_queue()
        mock_git_ops = MagicMock()
        mock_git_ops.project_root = tmp_path
        worker = SpeculativeMergeWorker(
            mock_git_ops, asyncio.Queue(), escalation_queue=escalation_spy,
        )

        item = _make_spec_item(tmp_path, cfg, speculative=True)

        # Unparseable: output has a nextest Summary footer (N=1 test ran) but no
        # per-test lines that parse_per_test_results can match.  This triggers
        # _alarm_warm_shadow_unparseable: _nextest_reported_test_count returns 1
        # (not None/0) → format-mismatch alarm fires (κ inv.6 fail-closed).
        # Using 'Build succeeded' + Summary but no PASS/FAIL lines ensures
        # parse_per_test_results returns {} while the discriminator sees N>0.
        unparseable_vr = VerifyResult(
            passed=True,
            test_output=(
                'Build succeeded\n'
                '        Summary [   0.01s] 1 test run: 1 passed, 0 failed, 0 skipped\n'
            ),
            lint_output='', type_output='', summary='',
        )

        async def _mock_verify_unparseable(*args, **kwargs):
            on_result = kwargs.get('on_result')
            if on_result is not None:
                on_result(unparseable_vr)
            return None

        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)

        with patch(
            'orchestrator.merge_queue._acquire_warm_verify_worktree',
            new=AsyncMock(return_value=(fake_lane, True)),
        ), patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=_mock_verify_unparseable,
        ):
            await worker._run_inflight_verify(item, _make_local_lease())

        # Unparseable warm output must trigger the fail-closed alarm
        # (κ invariant 6: never a silent pass)
        escalation_spy.submit.assert_called_once()


# ===========================================================================
# Step-19: RED — B9 capstone integration test
# ===========================================================================


def _make_b9_worker(tmp_path: Path) -> tuple[SpeculativeMergeWorker, MagicMock]:
    """Build a SpeculativeMergeWorker with a spy git_ops for B9 integration tests.

    The git_ops mock is pre-wired with:
    - acquire_spec_lane: side_effect list of ``(fake_lane, True)`` tuples so
      sequential calls return distinct lanes.
    - release_spec_lane / cleanup_merge_worktree: AsyncMock spies.
    - advance_main: returns 'advanced' immediately.

    Returns ``(worker, mock_git_ops)`` so tests can inspect call counts.
    """
    mock_git_ops = MagicMock()
    mock_git_ops.project_root = tmp_path

    fake_lane_0 = tmp_path / '_spec-0'
    fake_lane_1 = tmp_path / '_spec-1'
    for lane in (fake_lane_0, fake_lane_1):
        lane.mkdir(parents=True, exist_ok=True)

    mock_git_ops.acquire_spec_lane = AsyncMock(
        side_effect=[(fake_lane_0, True), (fake_lane_1, True)],
    )
    mock_git_ops.release_spec_lane = AsyncMock()
    mock_git_ops.cleanup_merge_worktree = AsyncMock()
    mock_git_ops.advance_main = AsyncMock(return_value='advanced')

    worker = SpeculativeMergeWorker(mock_git_ops, asyncio.Queue())
    worker._host_allocator = None  # skip lease release in finally
    worker.VERIFY_ABANDON_POLL_SECS = 0.01  # fast poll for tests

    return worker, mock_git_ops


def _build_entry(
    item: SpeculativeItem,
    vr: InflightVerifyResult,
    *,
    merge_wt: Path,
) -> InflightEntry:
    """Wrap *vr* in an InflightEntry whose verify_task resolves to it immediately."""

    async def _resolved() -> InflightVerifyResult:
        return vr

    return InflightEntry(
        item=item,
        lease=None,
        verify_task=asyncio.ensure_future(_resolved()),
        merge_wt=merge_wt,
        was_speculative=True,
        phase='verifying',
    )


@pytest.mark.asyncio
class TestB9CapstoneIntegration:
    """B9 capstone: K>1 spec verifies WARM + distinct lanes + serial CAS.

    RED before step 20:
    - ``InflightVerifyResult`` has no ``spec_warm`` field.
    - ``_finalize_inflight`` calls ``cleanup_merge_worktree`` on the spec lane
      path instead of ``release_spec_lane``; the lane is deleted rather than
      retained for reuse.

    After step 20 (GREEN):
    - ``spec_warm`` threaded through ``InflightVerifyResult``.
    - ``_finalize_inflight`` routes warm spec lanes to ``release_spec_lane``
      (retaining target/) and leaves non-spec paths byte-identical.
    """

    async def test_spec_lanes_released_not_cleaned_after_pass(self, tmp_path: Path):
        """After two spec verifies pass + finalize, release_spec_lane called for each.

        RED: ``_finalize_inflight`` unconditionally calls ``cleanup_merge_worktree``
        (which removes the lane from disk).  ``release_spec_lane`` is never called →
        assertion fails.

        GREEN (step 20): ``spec_warm=True`` threaded through ``InflightVerifyResult``;
        ``_finalize_inflight`` routes to ``release_spec_lane(merge_wt, warm=True)``
        so the lane is retained (target/ warmth preserved for the next candidate).
        """
        cfg = OrchestratorConfig(
            project_root=tmp_path,
            git=_make_spec_git_config(on=True),
        )
        worker, mock_git_ops = _make_b9_worker(tmp_path)

        fake_lane_0 = tmp_path / '_spec-0'
        fake_lane_1 = tmp_path / '_spec-1'

        item0 = _make_spec_item(tmp_path, cfg, speculative=True)
        item0.request.task_id = 'task-b9-0'
        item1 = _make_spec_item(tmp_path, cfg, speculative=True)
        item1.request.task_id = 'task-b9-1'

        fake_vr = VerifyResult(
            passed=True,
            test_output='        PASS [0.01s] reify-spec test_b9\n',
            lint_output='', type_output='', summary='',
        )

        async def _mock_verify(*args, **kwargs):
            on_result = kwargs.get('on_result')
            if on_result is not None:
                on_result(fake_vr)
            return None

        lease = _make_local_lease()

        with patch('orchestrator.merge_queue._run_post_merge_verify', side_effect=_mock_verify):
            vr0 = await worker._run_inflight_verify(item0, lease)
            vr1 = await worker._run_inflight_verify(item1, lease)

        # Verify the acquire routing produced distinct lanes
        assert vr0.merge_wt == fake_lane_0, (
            f'Expected vr0.merge_wt={fake_lane_0!r}, got {vr0.merge_wt!r}; '
            'acquire_spec_lane must return distinct lanes for each spec item'
        )
        assert vr1.merge_wt == fake_lane_1, (
            f'Expected vr1.merge_wt={fake_lane_1!r}, got {vr1.merge_wt!r}'
        )

        # Build InflightEntry wrappers and finalize in submission order
        entry0 = _build_entry(item0, vr0, merge_wt=fake_lane_0)
        entry1 = _build_entry(item1, vr1, merge_wt=fake_lane_1)

        done_outcome = MergeOutcome('done', merge_sha='newsha0000000000000a')
        with (
            patch(
                'orchestrator.merge_queue._finalize_advanced_merge',
                new=AsyncMock(return_value=done_outcome),
            ),
            patch('orchestrator.merge_queue._maybe_schedule_shadow_compare', new=AsyncMock()),
            patch('orchestrator.merge_queue._maybe_run_drift_check', new=AsyncMock()),
        ):
            await worker._finalize_inflight(entry0)
            await worker._finalize_inflight(entry1)

        # RED: release_spec_lane not called (cleanup_merge_worktree called instead)
        # GREEN: release_spec_lane called once per warm spec lane
        assert mock_git_ops.release_spec_lane.call_count == 2, (
            f'Expected release_spec_lane called 2 times (warm lane retained); '
            f'got {mock_git_ops.release_spec_lane.call_count}. '
            'After step 20, _finalize_inflight must route spec lanes to '
            'release_spec_lane rather than cleanup_merge_worktree.'
        )
        # cleanup_merge_worktree must NOT have been called on spec lane paths
        cleaned = [
            c.args[0]
            for c in mock_git_ops.cleanup_merge_worktree.call_args_list
        ]
        assert fake_lane_0 not in cleaned, (
            f'cleanup_merge_worktree was called on spec lane {fake_lane_0} — '
            'step 20 must route warm spec lanes to release_spec_lane instead'
        )
        assert fake_lane_1 not in cleaned, (
            f'cleanup_merge_worktree was called on spec lane {fake_lane_1} — '
            'step 20 must route warm spec lanes to release_spec_lane instead'
        )

    async def test_advance_main_submission_order_regression_guard(self, tmp_path: Path):
        """advance_main called in submission order — CAS serial ordering preserved at K>1.

        Regression guard: Lever C's single-Verifier serial finalize contract must
        hold even when two speculative items verified warm concurrently.  Finalizing
        entry0 then entry1 must call advance_main for item0 (expected_main=base_sha_0)
        before item1 (expected_main=base_sha_1).
        """
        cfg = OrchestratorConfig(
            project_root=tmp_path,
            git=_make_spec_git_config(on=True),
        )
        worker, mock_git_ops = _make_b9_worker(tmp_path)

        # Give items distinct base_shas so we can verify ordering
        item0 = _make_spec_item(tmp_path, cfg, speculative=True)
        item0.request.task_id = 'task-b9-order-0'
        # Override base_sha to distinguish the two advance_main calls
        from dataclasses import replace as _replace
        item0 = _replace(item0, base_sha='aaaa000000000000aaaa')

        item1 = _make_spec_item(tmp_path, cfg, speculative=True)
        item1.request.task_id = 'task-b9-order-1'
        item1 = _replace(item1, base_sha='bbbb000000000000bbbb')

        fake_vr = VerifyResult(
            passed=True,
            test_output='        PASS [0.01s] reify-spec test_order\n',
            lint_output='', type_output='', summary='',
        )

        async def _mock_verify(*args, **kwargs):
            on_result = kwargs.get('on_result')
            if on_result is not None:
                on_result(fake_vr)
            return None

        lease = _make_local_lease()

        with patch('orchestrator.merge_queue._run_post_merge_verify', side_effect=_mock_verify):
            vr0 = await worker._run_inflight_verify(item0, lease)
            vr1 = await worker._run_inflight_verify(item1, lease)

        entry0 = _build_entry(item0, vr0, merge_wt=tmp_path / '_spec-0')
        entry1 = _build_entry(item1, vr1, merge_wt=tmp_path / '_spec-1')

        done_outcome = MergeOutcome('done', merge_sha='newsha0000000000000a')
        with (
            patch(
                'orchestrator.merge_queue._finalize_advanced_merge',
                new=AsyncMock(return_value=done_outcome),
            ),
            patch('orchestrator.merge_queue._maybe_schedule_shadow_compare', new=AsyncMock()),
            patch('orchestrator.merge_queue._maybe_run_drift_check', new=AsyncMock()),
        ):
            # Submission order: entry0 first, entry1 second
            await worker._finalize_inflight(entry0)
            await worker._finalize_inflight(entry1)

        advance_calls = mock_git_ops.advance_main.call_args_list
        assert len(advance_calls) == 2, (
            f'Expected 2 advance_main calls, got {len(advance_calls)}'
        )
        # First call must use item0's base_sha (submission order preserved)
        # advance_main(current_sha, merge_wt, branch=..., max_attempts=..., expected_main=...)
        # Use kwargs since it's called as a keyword argument
        call0_kwargs = advance_calls[0].kwargs
        call1_kwargs = advance_calls[1].kwargs
        assert call0_kwargs.get('expected_main') == 'aaaa000000000000aaaa', (
            f"First advance_main call must use item0.base_sha='aaaa000000000000aaaa'; "
            f"got expected_main={call0_kwargs.get('expected_main')!r}"
        )
        assert call1_kwargs.get('expected_main') == 'bbbb000000000000bbbb', (
            f"Second advance_main call must use item1.base_sha='bbbb000000000000bbbb'; "
            f"got expected_main={call1_kwargs.get('expected_main')!r}"
        )

    async def test_warm_results_non_empty_feed_shadow_via_finalize(self, tmp_path: Path):
        """_maybe_schedule_shadow_compare receives non-empty warm_results via finalize.

        Confirms that the spec-lane warm verify populates warm_results (step 16
        wired _is_warm_path for spec lanes) AND that _finalize_inflight threads
        those results through to _maybe_schedule_shadow_compare (already wired
        at merge_queue.py:8139) — completing the end-to-end shadow valve path
        for speculative verifies.
        """
        cfg = OrchestratorConfig(
            project_root=tmp_path,
            git=_make_spec_git_config(
                on=True,
                warm_verify_shadow_compare=True,
                warm_verify_shadow_compare_every_n_merges=1,
            ),
        )
        worker, mock_git_ops = _make_b9_worker(tmp_path)

        item = _make_spec_item(tmp_path, cfg, speculative=True)
        item.request.task_id = 'task-b9-shadow'

        fake_vr = VerifyResult(
            passed=True,
            test_output='        PASS [0.01s] reify-spec test_b9_shadow\n',
            lint_output='', type_output='', summary='',
        )

        async def _mock_verify(*args, **kwargs):
            on_result = kwargs.get('on_result')
            if on_result is not None:
                on_result(fake_vr)
            return None

        lease = _make_local_lease()

        with patch('orchestrator.merge_queue._run_post_merge_verify', side_effect=_mock_verify):
            vr = await worker._run_inflight_verify(item, lease)

        # warm_results must be non-empty (step 16 wired _is_warm_path for spec lanes)
        assert vr.warm_results, (
            f'warm_results must be non-empty for spec-lane warm verify; '
            f'got {vr.warm_results!r}'
        )

        # Now confirm that _finalize_inflight threads warm_results to shadow valve
        shadow_compare_spy = AsyncMock()
        entry = _build_entry(item, vr, merge_wt=tmp_path / '_spec-0')

        done_outcome = MergeOutcome('done', merge_sha='newsha0000000000000a')
        with (
            patch(
                'orchestrator.merge_queue._finalize_advanced_merge',
                new=AsyncMock(return_value=done_outcome),
            ),
            patch(
                'orchestrator.merge_queue._maybe_schedule_shadow_compare',
                new=shadow_compare_spy,
            ),
            patch('orchestrator.merge_queue._maybe_run_drift_check', new=AsyncMock()),
        ):
            await worker._finalize_inflight(entry)

        # Shadow compare must have been called (outcome='done' triggers it)
        shadow_compare_spy.assert_called_once()
        # And the warm_results kwarg must match the non-empty results from the spec-lane verify
        call_kwargs = shadow_compare_spy.call_args.kwargs
        assert call_kwargs.get('warm_results'), (
            f'_maybe_schedule_shadow_compare must receive non-empty warm_results; '
            f'got warm_results={call_kwargs.get("warm_results")!r}'
        )


# ===========================================================================
# Step-21: RED — abort paths in _run_inflight_verify must RELEASE spec lanes
# ===========================================================================


def _make_abort_test_worker(
    tmp_path: Path,
) -> tuple[SpeculativeMergeWorker, MagicMock]:
    """Build a SpeculativeMergeWorker with async release/cleanup spies for abort tests.

    Sets VERIFY_ABANDON_POLL_SECS=0.01 so the abort-poll loop ticks fast and
    the test can trigger abort conditions after a short asyncio.sleep.
    """
    mock_git_ops = MagicMock()
    mock_git_ops.project_root = tmp_path
    mock_git_ops.release_spec_lane = AsyncMock()
    mock_git_ops.cleanup_merge_worktree = AsyncMock()
    worker = SpeculativeMergeWorker(mock_git_ops, asyncio.Queue())
    worker.VERIFY_ABANDON_POLL_SECS = 0.01
    return worker, mock_git_ops


@pytest.mark.asyncio
class TestSpecLaneAbortPathRelease:
    """_run_inflight_verify abort paths (DROPPED / REQUEUED) must release spec lanes.

    RED: both abort branches at merge_queue.py:7838 (DROPPED) and 7857 (REQUEUED)
    call ``_cleanup_owned_merge_worktree(merge_wt)`` unconditionally.  When
    ``merge_wt`` is a warm _spec- lane (``_spec_warm=True``), this removes the
    lane from disk via ``git worktree remove``, leaving WarmLanePool ASSIGNED
    forever — a permanent pool leak.

    GREEN (step-22): both branches call
    ``_release_or_cleanup(merge_wt, spec_warm=_spec_warm)`` which routes to
    ``release_spec_lane(lane, warm=True)`` for the warm spec-lane path.
    """

    async def test_dropped_abort_releases_spec_lane(self, tmp_path: Path):
        """DROPPED abort (sole-waiter cancelled) must release_spec_lane, not cleanup.

        Setup: verify hangs; req.result cancelled → _request_abandoned → DROPPED.
        Assert: release_spec_lane(fake_lane, warm=True) awaited;
                cleanup_merge_worktree NOT called on the spec lane.

        RED: _cleanup_owned_merge_worktree called unconditionally → spec lane removed.
        GREEN (step-22): _release_or_cleanup routes spec lane to release_spec_lane.
        """
        cfg = OrchestratorConfig(project_root=tmp_path, git=_make_spec_git_config(on=True))
        worker, mock_git_ops = _make_abort_test_worker(tmp_path)

        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)
        item = _make_spec_item(tmp_path, cfg, speculative=True)
        lease = _make_local_lease()

        async def _hang_verify(*args, **kwargs):  # noqa: ARG001
            await asyncio.sleep(100)
            return None

        with patch(
            'orchestrator.merge_queue._acquire_warm_verify_worktree',
            new=AsyncMock(return_value=(fake_lane, True)),
        ), patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=_hang_verify,
        ):
            inflight_task = asyncio.ensure_future(
                worker._run_inflight_verify(item, lease)
            )
            # Let the poll loop spin (>= 5 ticks at VERIFY_ABANDON_POLL_SECS=0.01)
            await asyncio.sleep(0.08)
            # Cancel result future → triggers _request_abandoned → DROPPED path
            item.request.result.cancel()
            result = await inflight_task

        assert result.status == 'DROPPED', (
            f'Expected DROPPED, got status={result.status!r}'
        )
        # GREEN: warm spec lane must be RELEASED, not git-worktree-removed
        mock_git_ops.release_spec_lane.assert_called_once_with(fake_lane, warm=True)
        # cleanup_merge_worktree must NOT be called on the spec lane path
        cleaned = [c.args[0] for c in mock_git_ops.cleanup_merge_worktree.call_args_list]
        assert fake_lane not in cleaned, (
            f'cleanup_merge_worktree was called on spec lane {fake_lane!r} — '
            'DROPPED abort path must route warm _spec- lanes to release_spec_lane '
            '(permanent pool leak otherwise)'
        )

    async def test_requeued_abort_releases_spec_lane(self, tmp_path: Path):
        """REQUEUED abort (operator halt) must release_spec_lane, not cleanup.

        Setup: verify hangs; worker._operator_halt set → REQUEUED (req re-queued).
        Assert: release_spec_lane(fake_lane, warm=True) awaited;
                cleanup_merge_worktree NOT called on the spec lane.

        RED: _cleanup_owned_merge_worktree called unconditionally → spec lane removed.
        GREEN (step-22): _release_or_cleanup routes spec lane to release_spec_lane.
        """
        cfg = OrchestratorConfig(project_root=tmp_path, git=_make_spec_git_config(on=True))
        worker, mock_git_ops = _make_abort_test_worker(tmp_path)

        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)
        item = _make_spec_item(tmp_path, cfg, speculative=True)
        lease = _make_local_lease()

        async def _hang_verify(*args, **kwargs):  # noqa: ARG001
            await asyncio.sleep(100)
            return None

        with patch(
            'orchestrator.merge_queue._acquire_warm_verify_worktree',
            new=AsyncMock(return_value=(fake_lane, True)),
        ), patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=_hang_verify,
        ):
            inflight_task = asyncio.ensure_future(
                worker._run_inflight_verify(item, lease)
            )
            # Let the poll loop spin
            await asyncio.sleep(0.08)
            # Set operator halt → triggers REQUEUED path
            worker._operator_halt.set()
            result = await inflight_task

        assert result.status == 'REQUEUED', (
            f'Expected REQUEUED, got status={result.status!r}'
        )
        # GREEN: warm spec lane must be RELEASED, not git-worktree-removed
        mock_git_ops.release_spec_lane.assert_called_once_with(fake_lane, warm=True)
        # cleanup_merge_worktree must NOT be called on the spec lane path
        cleaned = [c.args[0] for c in mock_git_ops.cleanup_merge_worktree.call_args_list]
        assert fake_lane not in cleaned, (
            f'cleanup_merge_worktree was called on spec lane {fake_lane!r} — '
            'REQUEUED abort path must route warm _spec- lanes to release_spec_lane '
            '(permanent pool leak otherwise)'
        )


# ===========================================================================
# Step-23: RED — _finalize_inflight terminal paths must RELEASE spec lanes
# ===========================================================================


def _make_finalize_test_worker(
    tmp_path: Path,
) -> tuple[SpeculativeMergeWorker, MagicMock]:
    """Build a SpeculativeMergeWorker with async spies for finalize terminal-path tests.

    Sets side-channel attributes needed by the rebased_pending_reverify path
    (``_last_advanced_sha``, ``_rebased_from``, ``_rebased_onto``) so the gate-
    retry test can trigger that path without real git I/O.
    """
    mock_git_ops = MagicMock()
    mock_git_ops.project_root = tmp_path
    mock_git_ops.release_spec_lane = AsyncMock()
    mock_git_ops.cleanup_merge_worktree = AsyncMock()
    mock_git_ops.advance_main = AsyncMock()
    mock_git_ops.get_main_sha = AsyncMock(return_value='deadbeef' * 5)
    # Side-channel attrs read by the rebased_pending_reverify path
    mock_git_ops._last_advanced_sha = 'aa' * 20
    mock_git_ops._rebased_from = 'bb' * 20
    mock_git_ops._rebased_onto = 'cc' * 20
    worker = SpeculativeMergeWorker(mock_git_ops, asyncio.Queue())
    return worker, mock_git_ops


@pytest.mark.asyncio
class TestSpecLaneFinalizeTerminalRelease:
    """_finalize_inflight non-advanced terminal paths must release spec lanes, not remove.

    RED: all four terminal cleanup sites call
    ``_cleanup_owned_merge_worktree(merge_wt)`` unconditionally:
      (a) gate-retry-exhausted     ~8226
      (b) HALT_ADVANCE_RESULTS + request-abandoned  ~8263
      (c) advance-failure non-CAS  ~8270
      (d) CAS-retry-exhausted      ~8294

    When ``merge_wt`` is a warm _spec- lane (``_vr_spec_warm=True``),
    ``cleanup_merge_worktree`` destroys the lane while WarmLanePool still has
    it ASSIGNED → permanent pool leak.

    GREEN (step-24): all four sites call
    ``_release_or_cleanup(merge_wt, spec_warm=_vr_spec_warm)`` which routes
    to ``release_spec_lane(lane, warm=True)`` for warm spec lanes.
    """

    async def test_gate_retry_exhausted_releases_spec_lane(self, tmp_path: Path):
        """Gate-retry-exhausted path (~8226) must release_spec_lane for a warm lane.

        Drive: advance_main returns 'rebased_pending_reverify' > MAX_CAS_RETRIES=5
        times; _reverify_rebased_tree returns None (gate clears each iteration).
        After the 6th gate attempt gate_total=6 > 5 → exhausted path fires.

        RED: _cleanup_owned_merge_worktree(merge_wt) called → spec lane removed.
        GREEN (step-24): _release_or_cleanup routes to release_spec_lane.
        """
        cfg = OrchestratorConfig(project_root=tmp_path, git=_make_spec_git_config(on=True))
        worker, mock_git_ops = _make_finalize_test_worker(tmp_path)

        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)

        item = _make_spec_item(tmp_path, cfg, speculative=True)
        item.request.task_id = 'task-gate-exhaust'

        # vr with spec_warm=True so _vr_spec_warm is True in _finalize_inflight
        vr = InflightVerifyResult(outcome=None, merge_wt=fake_lane, spec_warm=True)
        entry = _build_entry(item, vr, merge_wt=fake_lane)

        # Always return 'rebased_pending_reverify' to spin the gate-retry loop
        mock_git_ops.advance_main.return_value = 'rebased_pending_reverify'

        with patch(
            'orchestrator.merge_queue._reverify_rebased_tree',
            new=AsyncMock(return_value=None),  # gate clears every iteration
        ):
            await worker._finalize_inflight(entry)

        # After 6 gate-retry iterations (gate_total=6 > MAX_CAS_RETRIES=5):
        # GREEN: release_spec_lane(fake_lane, warm=True) called once
        mock_git_ops.release_spec_lane.assert_called_once_with(fake_lane, warm=True)
        cleaned = [c.args[0] for c in mock_git_ops.cleanup_merge_worktree.call_args_list]
        assert fake_lane not in cleaned, (
            'gate-retry-exhausted path must call release_spec_lane, '
            'not cleanup_merge_worktree on warm _spec- lane'
        )

    async def test_halt_abandoned_releases_spec_lane(self, tmp_path: Path):
        """HALT_ADVANCE_RESULTS + request-abandoned path (~8263) must release_spec_lane.

        Drive: advance_main side-effect cancels req.result then returns 'wip_overlap'
        (in _HALT_ADVANCE_RESULTS) so the short-circuit check (8099) is bypassed
        but the CAS-loop halt-abandoned branch fires.

        RED: _cleanup_owned_merge_worktree called → spec lane removed.
        GREEN (step-24): _release_or_cleanup routes to release_spec_lane.
        """
        cfg = OrchestratorConfig(project_root=tmp_path, git=_make_spec_git_config(on=True))
        worker, mock_git_ops = _make_finalize_test_worker(tmp_path)

        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)

        item = _make_spec_item(tmp_path, cfg, speculative=True)
        item.request.task_id = 'task-halt-abandoned'

        vr = InflightVerifyResult(outcome=None, merge_wt=fake_lane, spec_warm=True)
        entry = _build_entry(item, vr, merge_wt=fake_lane)

        req = item.request

        async def _advance_and_cancel(*args, **kwargs):  # noqa: ARG001
            # Cancel req.result DURING advance_main so _request_abandoned is
            # True in the CAS loop but False at the pre-loop short-circuit (8099).
            req.result.cancel()
            return 'wip_overlap'

        mock_git_ops.advance_main.side_effect = _advance_and_cancel

        await worker._finalize_inflight(entry)

        mock_git_ops.release_spec_lane.assert_called_once_with(fake_lane, warm=True)
        cleaned = [c.args[0] for c in mock_git_ops.cleanup_merge_worktree.call_args_list]
        assert fake_lane not in cleaned, (
            'halt-abandoned path must call release_spec_lane, '
            'not cleanup_merge_worktree on warm _spec- lane'
        )

    async def test_advance_failure_non_cas_releases_spec_lane(self, tmp_path: Path):
        """Advance-failure non-CAS path (~8270) must release_spec_lane for a warm lane.

        Drive: advance_main returns 'merge_conflict' (not in _HALT_ADVANCE_RESULTS,
        not 'cas_failed', not 'advanced', not 'rebased_pending_reverify') so the
        ``if result != 'cas_failed':`` branch fires; _map_advance_failure patched.

        RED: _cleanup_owned_merge_worktree called → spec lane removed.
        GREEN (step-24): _release_or_cleanup routes to release_spec_lane.
        """
        cfg = OrchestratorConfig(project_root=tmp_path, git=_make_spec_git_config(on=True))
        worker, mock_git_ops = _make_finalize_test_worker(tmp_path)

        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)

        item = _make_spec_item(tmp_path, cfg, speculative=True)
        item.request.task_id = 'task-advance-fail'

        vr = InflightVerifyResult(outcome=None, merge_wt=fake_lane, spec_warm=True)
        entry = _build_entry(item, vr, merge_wt=fake_lane)

        # 'merge_conflict' is not in _HALT_ADVANCE_RESULTS and not 'cas_failed'
        mock_git_ops.advance_main.return_value = 'merge_conflict'

        with patch(
            'orchestrator.merge_queue._map_advance_failure',
            new=AsyncMock(return_value=MergeOutcome('blocked', reason='conflict-test')),
        ):
            await worker._finalize_inflight(entry)

        mock_git_ops.release_spec_lane.assert_called_once_with(fake_lane, warm=True)
        cleaned = [c.args[0] for c in mock_git_ops.cleanup_merge_worktree.call_args_list]
        assert fake_lane not in cleaned, (
            'advance-failure non-CAS path must call release_spec_lane, '
            'not cleanup_merge_worktree on warm _spec- lane'
        )

    async def test_cas_retry_exhausted_releases_spec_lane(self, tmp_path: Path):
        """CAS-retry-exhausted path (~8294) must release_spec_lane for a warm lane.

        Drive: advance_main returns 'cas_failed' > MAX_CAS_RETRIES=5 times
        (total=6 > 5 → exhausted path fires).

        RED: _cleanup_owned_merge_worktree called → spec lane removed.
        GREEN (step-24): _release_or_cleanup routes to release_spec_lane.
        """
        cfg = OrchestratorConfig(project_root=tmp_path, git=_make_spec_git_config(on=True))
        worker, mock_git_ops = _make_finalize_test_worker(tmp_path)

        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)

        item = _make_spec_item(tmp_path, cfg, speculative=True)
        item.request.task_id = 'task-cas-exhaust'

        vr = InflightVerifyResult(outcome=None, merge_wt=fake_lane, spec_warm=True)
        entry = _build_entry(item, vr, merge_wt=fake_lane)

        # Always return 'cas_failed' to spin the CAS-retry loop
        mock_git_ops.advance_main.return_value = 'cas_failed'

        await worker._finalize_inflight(entry)

        # After 6 cas_failed iterations (total=6 > MAX_CAS_RETRIES=5):
        # GREEN: release_spec_lane(fake_lane, warm=True) called once
        mock_git_ops.release_spec_lane.assert_called_once_with(fake_lane, warm=True)
        cleaned = [c.args[0] for c in mock_git_ops.cleanup_merge_worktree.call_args_list]
        assert fake_lane not in cleaned, (
            'cas-retry-exhausted path must call release_spec_lane, '
            'not cleanup_merge_worktree on warm _spec- lane'
        )


# ===========================================================================
# Late-arrival integration scaffolding (pre-1 / task 1862)
# ===========================================================================
# Helpers for the "late-arrival attach to in-flight predecessor tip" tests
# (task 1862, steps 1–8).  The predecessor A's verify is gated so B is
# injected AFTER the Merger's one-shot look-ahead peek has run and found
# nothing — making B a genuine post-peek late arrival.
#
# Architecture overview
# ---------------------
# • spec_git_repo fixture (already defined above) provides the real repo.
# • _gated_runner / _inject_two_host_allocator / _make_branch_with_file /
#   _make_request are imported from test_merge_queue_concurrent_verify at the
#   top of this file (module-level, importable via conftest sys.path setup).
# • _LateArrivalFakeEventStore records speculative_merge / speculative_discard
#   events for assertion.
# • _make_late_arrival_git_config / _make_late_arrival_worker build the
#   worker.run() harness with K=2 (speculation_depth=2) and a wired
#   event_store.
#
# CONFTEST AUTOUSE OVERRIDE
# -------------------------
# The conftest autouse _mock_merge_queue_verification fixture monkeypatches
# orchestrator.merge_queue.run_scoped_verification to return passed=True.
# Each late-arrival integration test overrides this IN-BODY via:
#
#     with patch('orchestrator.merge_queue.run_scoped_verification', gated_local):
#
# A context-manager patch() applied inside the test body takes precedence
# over the function-scoped monkeypatch binding for the duration of the
# with-block — the same mechanism used by TestChainInvalidationUnderOverlap
# in test_merge_queue_concurrent_verify.py.
# ===========================================================================


class _LateArrivalFakeEventStore(EventStore):
    """Capturing EventStore for late-arrival speculative event assertions.

    Records every emit() call as a dict ``{'type', 'task_id', 'data'}``.
    Never performs SQLite I/O — ``object.__init__`` bypasses the DB setup
    in ``EventStore.__init__``.

    Usage::

        store = _LateArrivalFakeEventStore()
        worker = _make_late_arrival_worker(git_ops, event_store=store)
        ...
        spec_events = store.speculative_events(EventType.speculative_merge)
        assert spec_events[0]['data']['base_sha'] == expected_sha
    """

    def __init__(self) -> None:
        object.__init__(self)
        self.events: list[dict] = []

    def emit(  # type: ignore[override]
        self,
        event_type,
        *,
        task_id=None,
        phase=None,
        role=None,
        data=None,
        cost_usd=None,
        duration_ms=None,
    ) -> None:
        self.events.append({
            'type': event_type,
            'task_id': task_id,
            'data': data or {},
        })

    def speculative_events(
        self, event_type: EventType | None = None,
    ) -> list[dict]:
        """Return recorded events, optionally filtered by EventType."""
        if event_type is None:
            return list(self.events)
        return [e for e in self.events if e['type'] == event_type]


def _make_late_arrival_git_config() -> GitConfig:
    """Plain GitConfig for late-arrival integration tests (no warm-lane pool).

    Warm-lane pool is intentionally disabled so spec-lane routing does not
    interfere with the late-arrival mechanism under test.
    """
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


def _make_late_arrival_worker(
    git_ops: GitOps,
    event_store: _LateArrivalFakeEventStore | None = None,
    *,
    speculation_depth: int = 2,
) -> SpeculativeMergeWorker:
    """Build a K=2 SpeculativeMergeWorker wired for late-arrival integration tests.

    K=2 (speculation_depth=2) allows A and B to verify concurrently — A on
    the remote gated runner and B on the local slot once it attaches to
    A's pending spec-base.  The event_store is wired so speculative_merge /
    speculative_discard events are captured for DONE-WHEN 1/2/4 assertions.

    Returns the worker; the caller must inject a two-host HostAllocator via
    _inject_two_host_allocator(worker, gated_remote) before calling worker.run().
    """
    q: asyncio.Queue[MergeRequest] = asyncio.Queue()
    return SpeculativeMergeWorker(
        git_ops,
        q,
        event_store=event_store,
        speculation_depth=speculation_depth,
    )


# ===========================================================================
# Step-1 RED: late arrival attaches to in-flight predecessor's merge commit
# ===========================================================================


@pytest.mark.asyncio
class TestLateArrivalAttaches:
    """Step-1 RED — late arrival B attaches to in-flight predecessor A's merge commit.

    DONE-WHEN 1: a ``speculative_merge`` event is emitted for B with
    ``data.base_sha == A's merge commit SHA``.

    DONE-WHEN 2: ``git_ops.merge_to_main`` is invoked for B with
    ``base_sha == A's merge commit SHA``.

    On main (RED): line 6875 unconditionally resets ``spec_base = None`` on every
    fresh dequeue.  B is merged non-speculatively against plain main
    (``merge_to_main base_sha=None``); no ``speculative_merge`` event is emitted
    for B.

    GREEN after step-2: ``pending_spec_base`` carries A's merge commit across the
    blocking dequeue so B attaches and is merged against main+A.
    """

    async def test_late_arrival_attaches_to_predecessor_tip(
        self, spec_git_repo: Path,
    ) -> None:
        """Genuine post-peek late arrival B merges against A's in-flight commit.

        RED:
          - ``b_merge['base_sha']`` is ``None`` (non-speculative plain-main merge)
          - no ``speculative_merge`` event with B's task_id
        GREEN (step-2):
          - ``b_merge['base_sha'] == a_merge_commit``
          - ``speculative_merge`` event for B carries the same ``base_sha``
        """
        git_config = _make_late_arrival_git_config()
        git_ops = GitOps(git_config, spec_git_repo)
        fake_event_store = _LateArrivalFakeEventStore()

        # ── Spy on merge_to_main to capture base_sha and merge commits ─────────
        _merge_calls: list[dict] = []
        original_merge_to_main = git_ops.merge_to_main

        async def _spy_merge_to_main(
            wt: Path, branch: str, base_sha: str | None = None,
        ) -> Any:
            result = await original_merge_to_main(wt, branch, base_sha=base_sha)
            _merge_calls.append({
                'branch': branch,
                'base_sha': base_sha,
                'merge_commit': result.merge_commit,
                'success': result.success,
            })
            return result

        git_ops.merge_to_main = _spy_merge_to_main  # type: ignore[method-assign]

        # ── Gate for A's LOCAL verify ──────────────────────────────────────────
        # Gating the LOCAL verify (run_scoped_verification) holds A in-flight so
        # the Merger's look-ahead peek fires and finds nothing, then blocks waiting
        # for a new request.  gate_a_entered fires inside _gated_local on the first
        # call — by then the look-ahead has already completed (it has no yield points
        # between permit-acquire and pop, so it runs synchronously before the verify
        # task gets an event-loop tick).
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        # ── B's remote runner — passes immediately (no gate needed for step-1) ─
        gate_b_prerelease = asyncio.Event()
        gate_b_prerelease.set()  # B's remote verify returns right away
        fake_remote = _gated_runner(
            gate_b_prerelease, passed=True, name='late-step1-laptop',
        )

        # ── Build branches ─────────────────────────────────────────────────────
        config = OrchestratorConfig(project_root=spec_git_repo, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/late1-a', 'late1_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/late1-b', 'late1_b.py', 'b = 2\n',
        )

        # ── Build K=2 worker ───────────────────────────────────────────────────
        worker = _make_late_arrival_worker(git_ops, event_store=fake_event_store)
        _inject_two_host_allocator(worker, fake_remote)

        req_a = _make_request('late1-a', 'task/late1-a', wt_a, config)
        req_b = _make_request('late1-b', 'task/late1-b', wt_b, config)

        # ── Run the harness ────────────────────────────────────────────────────
        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            # Enqueue A only — B is withheld until after the look-ahead peek.
            await worker._queue.put(req_a)

            # A's verify enters the gate.  By this point the Merger has already:
            #   (1) merged A and put it on the verifier queue
            #   (2) acquired the speculation permit
            #   (3) done the look-ahead peek (synchronous, no yield) → found nothing
            #   (4) released the permit, set spec_base=None
            #   (5) blocked in _acquire_next_request()
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)

            # B arrives LATE — after the one-shot look-ahead has already fired.
            await worker._queue.put(req_b)

            # Poll until B's merge_to_main call appears in the spy (indicates the
            # Merger has dequeued B and merged it — either speculatively or not).
            deadline = asyncio.get_running_loop().time() + 10.0
            while (
                not any(c['branch'] == 'task/late1-b' for c in _merge_calls)
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.05)

            # ── Assertions while A is still in-flight ─────────────────────────
            b_spy = [c for c in _merge_calls if c['branch'] == 'task/late1-b']
            a_spy = [c for c in _merge_calls if c['branch'] == 'task/late1-a']

            assert a_spy, 'merge_to_main must have been called for A'
            assert b_spy, 'merge_to_main must have been called for B (timed out waiting)'

            a_merge_commit = a_spy[0]['merge_commit']
            assert a_merge_commit is not None, 'A must have a non-None merge commit'
            a_merge_commit = a_merge_commit.strip()

            # DONE-WHEN 2: merge_to_main for B invoked with base_sha == A's merge commit.
            # RED: base_sha=None (line 6875 resets spec_base on fresh dequeue → B non-spec).
            # GREEN (step-2): base_sha == a_merge_commit (pending_spec_base attaches B).
            assert b_spy[0]['base_sha'] == a_merge_commit, (
                f'B must be merged speculatively against A\'s merge commit '
                f'{a_merge_commit!r}; got base_sha={b_spy[0]["base_sha"]!r}.\n'
                'RED: _merger_loop line 6875 resets spec_base=None on fresh dequeue '
                'so B is merged against plain main (base_sha=None).\n'
                'GREEN (step-2): pending_spec_base holds A\'s commit across the '
                'blocking _acquire_next_request → B.base_sha == A\'s merge commit.'
            )

            # DONE-WHEN 1: speculative_merge event for B with base_sha == A's merge commit.
            # RED: no speculative_merge event for B (B is non-speculative).
            b_spec_events = fake_event_store.speculative_events(EventType.speculative_merge)
            b_spec_events = [e for e in b_spec_events if e['task_id'] == 'late1-b']

            assert len(b_spec_events) == 1, (
                f'Expected exactly one speculative_merge event for B; '
                f'got {len(b_spec_events)}.\n'
                'RED: B is merged non-speculatively (spec_base=None) → no '
                'speculative_merge event emitted.\n'
                'GREEN (step-2): B attaches to A\'s pending spec base → '
                'speculative=True → speculative_merge emitted before merge_to_main.'
            )
            assert b_spec_events[0]['data'].get('base_sha') == a_merge_commit, (
                f'speculative_merge event for B must carry base_sha={a_merge_commit!r}; '
                f'got {b_spec_events[0]["data"]!r}.\n'
                'RED: event not emitted at all (B non-speculative).\n'
                'GREEN (step-2): base_sha = pending_spec_base = A\'s merge commit.'
            )

            # ── Release A's gate → let A and B complete cleanly ───────────────
            gate_a_release.set()

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(req_a.result, timeout=10.0)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(req_b.result, timeout=10.0)

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)


# ===========================================================================
# Step-3 RED→GREEN: predecessor landing produces clean CAS — no disjoint-skip
# ===========================================================================


@pytest.mark.asyncio
class TestLateArrivalCleanCAS:
    """Step-3 RED→GREEN — after A lands, B advances via clean CAS (DONE-WHEN 3).

    DONE-WHEN 3: _reverify_rebased_tree is NEVER called for B; advance_main for B
    returns 'advanced' (not 'rebased_pending_reverify'); both files are on main;
    B's merge SHA is a descendant of A's merge commit.

    On main (RED): B merged against plain main → B.base_sha = main-before-A.
    When A lands, main = A's merge commit ≠ main-before-A → advance_main rebases
    → 'rebased_pending_reverify' → _reverify_rebased_tree disjoint fast-path runs.

    GREEN after step-2: B.base_sha = A's merge commit == main-after-A → clean CAS
    → 'advanced' → _reverify_rebased_tree unreachable.
    """

    async def test_predecessor_land_no_reverify_clean_advance(
        self, spec_git_repo: Path,
    ) -> None:
        """A lands first; B advances cleanly with no rebase and no _reverify_rebased_tree.

        RED: _reverify_rebased_tree called for B (disjoint fast-path); advance returns
        'rebased_pending_reverify'.

        GREEN (step-2): B.base_sha == A's merge commit == main-after-A → 'advanced'.
        """
        from orchestrator.merge_queue import (
            _reverify_rebased_tree as _orig_reverify,
        )

        git_config = _make_late_arrival_git_config()
        git_ops = GitOps(git_config, spec_git_repo)
        fake_event_store = _LateArrivalFakeEventStore()

        # ── Spy: capture advance_main result per branch ───────────────────────
        advance_outcomes: dict[str, str] = {}
        original_advance_main = git_ops.advance_main

        async def _spy_advance_main(
            merge_sha: str,
            merge_worktree: Any = None,
            branch: str | None = None,
            **kwargs: Any,
        ) -> Any:
            result = await original_advance_main(
                merge_sha, merge_worktree, branch=branch, **kwargs,
            )
            if branch:
                advance_outcomes[branch] = str(result)
            return result

        git_ops.advance_main = _spy_advance_main  # type: ignore[method-assign]

        # ── Spy: track _reverify_rebased_tree calls per task_id ───────────────
        reverify_calls: list[str] = []

        async def _spy_reverify(
            git_ops_arg: Any,
            req_arg: Any,
            merge_wt: Any,
            **kwargs: Any,
        ) -> Any:
            reverify_calls.append(getattr(req_arg, 'task_id', '?'))
            return await _orig_reverify(git_ops_arg, req_arg, merge_wt, **kwargs)

        # ── Gate A's LOCAL verify ──────────────────────────────────────────────
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            return MagicMock(
                passed=True, summary='ok', test_output='ok',
                lint_output='', type_output='', category='',
                timed_out=False, verify_skipped=False,
            )

        # ── Gate B's REMOTE verify — hold until A has landed ─────────────────
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='late3-laptop',
        )

        # ── Build branches (disjoint files) ──────────────────────────────────
        config = OrchestratorConfig(project_root=spec_git_repo, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/late3-a', 'late3_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/late3-b', 'late3_b.py', 'b = 2\n',
        )

        worker = _make_late_arrival_worker(git_ops, event_store=fake_event_store)
        _inject_two_host_allocator(worker, gated_remote)

        req_a = _make_request('late3-a', 'task/late3-a', wt_a, config)
        req_b = _make_request('late3-b', 'task/late3-b', wt_b, config)

        # ── Run harness ────────────────────────────────────────────────────────
        with (
            patch('orchestrator.merge_queue.run_scoped_verification', _gated_local),
            patch('orchestrator.merge_queue._reverify_rebased_tree', _spy_reverify),
        ):
            worker_task = asyncio.create_task(worker.run())

            # Enqueue A only; wait for its verify to enter the gate (look-ahead done).
            await worker._queue.put(req_a)
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)

            # Inject B late — after the one-shot look-ahead peek.
            await worker._queue.put(req_b)

            # Wait for B's remote verify to enter (B has been merged + dispatched).
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # Release A → A's verify completes, A's advance_main fires, A lands.
            gate_a_release.set()
            outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            assert outcome_a.status == 'done', f'A must land cleanly; got {outcome_a!r}'

            # A has landed → main == A's merge commit.  Now release B.
            gate_b_release.set()
            outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # ── Assertions ────────────────────────────────────────────────────────

        assert outcome_b.status == 'done', (
            f'B must land cleanly after A; got {outcome_b!r}'
        )

        # DONE-WHEN 3(a): _reverify_rebased_tree NEVER called for B.
        # RED: B.base_sha = plain-main ≠ main-after-A → rebase → reverify called.
        # GREEN (step-2): B.base_sha = A's commit = main-after-A → clean CAS → unreachable.
        assert 'late3-b' not in reverify_calls, (
            f'_reverify_rebased_tree must NOT be called for B; '
            f'was called for task_ids: {reverify_calls!r}.\n'
            'RED: B.base_sha = plain-main → advance_main rebases → '
            'rebased_pending_reverify → _reverify_rebased_tree disjoint fast-path.\n'
            'GREEN (step-2): B.base_sha == A\'s merge commit == main-after-A → '
            'clean CAS → _reverify_rebased_tree unreachable.'
        )

        # DONE-WHEN 3(b): advance_main for B returned 'advanced' (clean CAS).
        # RED: advance returns 'rebased_pending_reverify'.
        # GREEN (step-2): advance returns 'advanced'.
        b_branch = 'task/late3-b'
        assert advance_outcomes.get(b_branch) == 'advanced', (
            f'advance_main for B must return \'advanced\'; '
            f'got {advance_outcomes.get(b_branch)!r}.\n'
            'RED: B.base_sha = plain-main → mismatch with main-after-A → '
            '\'rebased_pending_reverify\'.\n'
            'GREEN (step-2): B.base_sha == A\'s merge commit → clean CAS → \'advanced\'.'
        )

        # DONE-WHEN 3(c): both files on main.
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'], cwd=spec_git_repo,
        )
        assert 'late3_a.py' in main_files, 'A\'s file (late3_a.py) must be on main'
        assert 'late3_b.py' in main_files, 'B\'s file (late3_b.py) must be on main'

        # DONE-WHEN 3(d): B's merge SHA is a descendant of A's.
        # After a clean CAS, B's merge commit was parented on A's merge commit.
        a_advance_branch = 'task/late3-a'
        a_advanced = advance_outcomes.get(a_advance_branch) == 'advanced'
        assert a_advanced, (
            f'A must also have advanced cleanly; got {advance_outcomes.get(a_advance_branch)!r}'
        )
