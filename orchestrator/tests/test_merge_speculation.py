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
    SpeculativeItem,
    _acquire_warm_verify_worktree,
    _maybe_schedule_shadow_compare,
    _run_cold_shadow_verify,
    _submit_shadow_divergence_escalation,
)
from orchestrator.verify import VerifyResult  # noqa: F401
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
            test_output='        PASS [0.05s] spec::test_warm_shadow\n',
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
            test_output='        PASS [0.05s] spec::test_warm_shadow\n',
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
            return {'spec::test_warm_shadow': 'pass'}  # parity with warm

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
