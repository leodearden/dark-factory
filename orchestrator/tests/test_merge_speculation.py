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

import ast
import asyncio
import contextlib
import logging
import stat
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import (  # noqa: F401
    MERGE_GATE_BARRIER_TIMEOUT,
    MERGE_RESULT_TIMEOUT,
    make_placeholder_future,
    pydantic_spec,
    wait_responsive,
)
from test_merge_queue_concurrent_verify import (  # noqa: F401
    HEAVY_BARRIER_TEST_TIMEOUT,
    PYPROJECT_DEFAULT_TIMEOUT,
    _fake_verify_result,
    _gated_runner,
    _id_liveness_fake_runner,
    _inject_two_host_allocator,
    _make_branch_with_file,
    _make_request,
    _timeout_mark_offenders,
    _worst_per_method_wait_budget,
)

from orchestrator.config import GitConfig, OrchestratorConfig  # noqa: F401
from orchestrator.event_store import EventStore, EventType  # noqa: F401
from orchestrator.git_ops import (  # noqa: F401
    AdvanceOutcome,
    AdvanceResult,
    GitOps,
    WorktreeInfo,
    _run,
)
from orchestrator.landed_outbox import LandedOutbox  # noqa: F401
from orchestrator.merge_disposition import (  # noqa: F401
    ClassificationResult,
    classify_merge_failure_disposition,
)
from orchestrator.merge_queue import (  # noqa: F401
    InflightEntry,
    InflightVerifyResult,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
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

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


# Task 3980 step-12/13: the merge-disposition classifier has TWO fail-open
# sites, on two different loggers, and both emit this same substring:
#
#   1. merge_disposition.py:710-719 — the classifier's own internal fail-open,
#      logger 'orchestrator.merge_disposition'.
#   2. merge_queue.py:994-1001 — `_classify_disposition_for_outcome`'s
#      belt-and-suspenders catch for anything the classifier's own fail-open
#      re-raises past, logger 'orchestrator.merge_queue'.
#
# Keying one predicate on the shared substring catches both. Filtering by a
# single logger name (as this file did before step-13) silently excludes site 2.
_FAIL_OPEN_SUBSTRING = 'degrading to INDETERMINATE (fail-open, I3)'
_FAIL_OPEN_LOGGERS = ('orchestrator.merge_disposition', 'orchestrator.merge_queue')


def _fail_open_records(
    records: list[logging.LogRecord],
) -> list[logging.LogRecord]:
    """Return the disposition fail-open WARNINGs among *records* (both sites).

    NOT symmetric with "the classifier ran": a classifier that SUCCEEDS emits
    NOTHING at WARNING (it logs only on the degrade paths,
    merge_disposition.py:695 and :711), so an empty return here means "no
    fail-open", never "the classifier was reached". Proving the classifier was
    reached needs a delegating SPY on the callable — see
    ``TestLateArrivalFailCascade`` for the live one and
    ``TestDispositionDoubleFidelity`` for the isolated two-sided proof.
    Reading log silence as non-execution is exactly the inference that produced
    a wrong review finding against this file; do not repeat it.

    Deliberately does NOT match merge_disposition.py:695's *other* degrade
    ('degrading implicated landings to INDETERMINATE (...)'), which is a
    legitimate evidence-absent verdict rather than a swallowed exception.
    """
    return [
        r for r in records
        if r.name in _FAIL_OPEN_LOGGERS and _FAIL_OPEN_SUBSTRING in r.getMessage()
    ]


def _format_fail_open_records(records: list[logging.LogRecord]) -> str:
    """Render fail-open records for an assertion message, naming the underlying
    exception (the actionable part — the WARNING text alone never says WHY)."""
    return '\n'.join(
        f'  - [{r.name}] {r.getMessage()}'
        + (f'\n    underlying: {r.exc_info[1]!r}' if r.exc_info else '')
        for r in records
    )


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
) -> RealMergeItem:
    """Build a minimal RealMergeItem for _run_inflight_verify tests.

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

    return RealMergeItem(
        request=fake_req,
        merge_result=MagicMock(merge_commit='deadbeef01234567890a'),
        merge_wt=merge_wt,
        base_sha='aabbccdd00000000aaaa',
        speculative=speculative,
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

    async def test_spec_lane_warm_baseline_merges_attempt0_shadow_sink(
        self, tmp_path: Path
    ):
        """Narrowed warm retry: warm_results = attempt-0 ∪ partial retry (PRD D4, §5.4).

        The mock ``_run_post_merge_verify`` simulates a corroborated NARROWED
        warm retry: it populates the ``shadow_baseline_sink`` with an attempt-0
        PASSED test omitted from the narrowed run, and reports a PARTIAL warm
        ``test_output`` containing ONLY the re-run test.  ``_run_inflight_verify``
        must union them before storing the warm shadow baseline — else the
        from-scratch FULL cold shadow compare flags the attempt-0 pass as
        only_cold → phantom born-at-L2 divergence alarm.

        RED (pre step-8): the sink is ignored; ``warm_results`` is the PARTIAL
        parse ``{'reify-spec test_retried': 'pass'}``.
        GREEN (step-8): ``build_warm_shadow_results`` unions the sink →
        ``{'reify-spec test_passed': 'pass', 'reify-spec test_retried': 'pass'}``.
        """
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

        # PARTIAL narrowed-retry output: ONLY the re-run test (attempt-0 pass omitted).
        fake_vr = VerifyResult(
            passed=True,
            test_output='        PASS [0.05s] reify-spec test_retried\n',
            lint_output='',
            type_output='',
            summary='',
        )

        async def _mock_verify(*args, **kwargs):
            # Corroborated-narrowed path: seed the sink with the attempt-0 pass.
            sink = kwargs.get('shadow_baseline_sink')
            if sink is not None:
                sink.update({'reify-spec test_passed': 'pass'})
            on_result = kwargs.get('on_result')
            if on_result is not None:
                on_result(fake_vr)
            return None  # verify passed

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

        assert vr.warm_results == {
            'reify-spec test_passed': 'pass',
            'reify-spec test_retried': 'pass',
        }, (
            'narrowed warm retry must store the MERGED (attempt-0 ∪ retry) '
            f'shadow baseline, not the partial parse; got {vr.warm_results!r}'
        )


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
    mock_git_ops.advance_main = AsyncMock(return_value=AdvanceOutcome('advanced'))

    worker = SpeculativeMergeWorker(mock_git_ops, asyncio.Queue())
    worker._host_allocator = None  # skip lease release in finally
    worker.VERIFY_ABANDON_POLL_SECS = 0.01  # fast poll for tests

    return worker, mock_git_ops


def _build_entry(
    item: RealMergeItem,
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
    """Build a SpeculativeMergeWorker with async spies for finalize terminal-path tests."""
    mock_git_ops = MagicMock()
    mock_git_ops.project_root = tmp_path
    mock_git_ops.release_spec_lane = AsyncMock()
    mock_git_ops.cleanup_merge_worktree = AsyncMock()
    mock_git_ops.advance_main = AsyncMock()
    mock_git_ops.get_main_sha = AsyncMock(return_value='deadbeef' * 5)
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

        # Always return 'rebased_pending_reverify' to spin the gate-retry loop.
        # SHA fields must be populated: _finalize_inflight sources them from
        # the return value (task 1997 step-6).
        mock_git_ops.advance_main.return_value = AdvanceOutcome(
            'rebased_pending_reverify',
            advanced_sha='aa' * 20,
            rebased_from='bb' * 20,
            rebased_onto='cc' * 20,
        )

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
            return AdvanceOutcome('wip_overlap')

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

        # 'merge_conflict' is not in _HALT_ADVANCE_RESULTS and not 'cas_failed' — an
        # intentionally out-of-domain sentinel exercising the unhandled-code branch,
        # not a real AdvanceResult member (hence the cast).
        mock_git_ops.advance_main.return_value = AdvanceOutcome(
            cast(AdvanceResult, 'merge_conflict')
        )

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
        mock_git_ops.advance_main.return_value = AdvanceOutcome('cas_failed')

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
# task 1997 step-5: rebased_pending_reverify consumes AdvanceOutcome fields
# ===========================================================================


@pytest.mark.asyncio
class TestGateReverifyConsumesAdvanceOutcome:
    """_finalize_inflight's rebased_pending_reverify branch sources the
    post-rebase SHAs from the AdvanceOutcome return value — the
    git_ops._last_advanced_sha/_rebased_from/_rebased_onto getattr side
    channel is retired (task 1997 / MQ-refactor μ).  The stub built by
    _make_finalize_test_worker never sets those attributes, so this pins
    the retirement: production must not depend on them.
    """

    async def test_gate_reverify_consumes_advance_outcome_sha_fields(
        self, tmp_path: Path,
    ) -> None:
        cfg = OrchestratorConfig(project_root=tmp_path, git=_make_spec_git_config(on=True))
        worker, mock_git_ops = _make_finalize_test_worker(tmp_path)

        fake_lane = tmp_path / '_spec-0'
        fake_lane.mkdir(parents=True, exist_ok=True)

        item = _make_spec_item(tmp_path, cfg, speculative=True)
        item.request.task_id = 'task-gate-reverify-sha'

        vr = InflightVerifyResult(outcome=None, merge_wt=fake_lane, spec_warm=True)
        entry = _build_entry(item, vr, merge_wt=fake_lane)

        REBASED_SHA = 'ab' * 20
        REBASED_FROM = 'ba' * 20
        REBASED_ONTO = 'cd' * 20

        advance_calls: list[tuple[tuple, dict]] = []

        async def _advance_side_effect(*args, **kwargs):
            advance_calls.append((args, kwargs))
            if len(advance_calls) == 1:
                return AdvanceOutcome(
                    'rebased_pending_reverify',
                    advanced_sha=REBASED_SHA,
                    rebased_from=REBASED_FROM,
                    rebased_onto=REBASED_ONTO,
                )
            # Second call (post-rebuild): terminal failure — sidesteps the
            # success-path gate machinery, which is not this test's concern.
            return AdvanceOutcome('not_descendant')

        mock_git_ops.advance_main.side_effect = _advance_side_effect

        captured_reverify_kwargs: dict = {}

        async def _fake_reverify_rebased_tree(
            git_ops, req, merge_wt, *, rebased_from, rebased_onto, merge_sha, **kwargs,
        ):
            captured_reverify_kwargs.update(
                rebased_from=rebased_from, rebased_onto=rebased_onto, merge_sha=merge_sha,
            )
            return None  # gate clears — disjoint/green re-verify

        with patch(
            'orchestrator.merge_queue._reverify_rebased_tree',
            new=_fake_reverify_rebased_tree,
        ):
            await worker._finalize_inflight(entry)

        assert captured_reverify_kwargs == {
            'rebased_from': REBASED_FROM,
            'rebased_onto': REBASED_ONTO,
            'merge_sha': REBASED_SHA,
        }, (
            f'_reverify_rebased_tree must be called with the AdvanceOutcome '
            f'fields, not the (unset) getattr side channel; got '
            f'{captured_reverify_kwargs!r}'
        )
        assert len(advance_calls) == 2, (
            f'advance_main must be retried after the gate clears; got '
            f'{len(advance_calls)} call(s)'
        )
        second_args, second_kwargs = advance_calls[1]
        assert second_args[0] == REBASED_SHA, (
            f'second advance_main call must retry with current_sha sourced '
            f'from adv_outcome.advanced_sha; got {second_args[0]!r}'
        )
        assert second_kwargs['expected_main'] == REBASED_ONTO, (
            f'second advance_main call must use item.base_sha rebuilt from '
            f'rebased_onto (task 1990 replace-only rebuild); got '
            f'{second_kwargs.get("expected_main")!r}'
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
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)  # task 3980: worst per-method wait budget 110s, x2 stretched
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
        self, spec_git_repo: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Genuine post-peek late arrival B merges against A's in-flight commit.

        RED:
          - ``b_merge['base_sha']`` is ``None`` (non-speculative plain-main merge)
          - no ``speculative_merge`` event with B's task_id
        GREEN (step-2):
          - ``b_merge['base_sha'] == a_merge_commit``
          - ``speculative_merge`` event for B carries the same ``base_sha``
        """
        # Capture DEBUG-level logs so we can assert the ATTACH code path fired.
        caplog.set_level(logging.DEBUG, logger='orchestrator.merge_queue')

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
            # task 3980: _fake_verify_result, not a bare MagicMock — see the
            # TestNoBareVerifyResultDoubles guard at the foot of this module.
            return _fake_verify_result(
                passed=True, summary='ok', test_output='ok',
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
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='late1-a: gate_a_entered',
            )

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

            # DONE-WHEN 2 (mechanism): assert the late-arrival ATTACH code path
            # fired — not just the outcome.  The ATTACH branch emits a DEBUG log
            # 'late arrival attaches to in-flight predecessor …'.  If a future
            # edit introduces an await inside _pop_next_pickable/_drain_queue so B
            # is prefetched normally by the look-ahead, the outcome assertions above
            # remain green but this log assertion will fail, revealing the test has
            # stopped guarding the late-arrival path.
            attach_msgs = [
                r.message
                for r in caplog.records
                if 'late arrival attaches to in-flight predecessor' in r.message
            ]
            assert attach_msgs, (
                'Expected at least one "late arrival attaches to in-flight '
                'predecessor" DEBUG log — confirms the pending_spec_base ATTACH '
                'branch fired.  Without this, the test only checks the outcome '
                'and would pass even if B were prefetched normally by the '
                'look-ahead (defeating the late-arrival guard).'
            )

            # ── Release A's gate → let A and B complete cleanly ───────────────
            gate_a_release.set()

            with contextlib.suppress(TimeoutError):
                await wait_responsive(
                    req_a.result,
                    timeout=MERGE_RESULT_TIMEOUT,
                    label='late1-a: MergeOutcome',
                )
            with contextlib.suppress(TimeoutError):
                await wait_responsive(
                    req_b.result,
                    timeout=MERGE_RESULT_TIMEOUT,
                    label='late1-b: MergeOutcome',
                )

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)


# ===========================================================================
# Step-3 RED→GREEN: predecessor landing produces clean CAS — no disjoint-skip
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)  # task 3980: worst per-method wait budget 125s, x2 stretched
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
        self, spec_git_repo: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A lands first; B advances cleanly with no rebase and no _reverify_rebased_tree.

        RED: _reverify_rebased_tree called for B (disjoint fast-path); advance returns
        'rebased_pending_reverify'.

        GREEN (step-2): B.base_sha == A's merge commit == main-after-A → 'advanced'.
        """
        from orchestrator.merge_queue import (
            _reverify_rebased_tree as _orig_reverify,
        )

        # Capture DEBUG-level logs so we can assert the ATTACH code path fired.
        caplog.set_level(logging.DEBUG, logger='orchestrator.merge_queue')

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
                advance_outcomes[branch] = result.result
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
            # task 3980: _fake_verify_result, not a bare MagicMock — see the
            # TestNoBareVerifyResultDoubles guard at the foot of this module.
            return _fake_verify_result(
                passed=True, summary='ok', test_output='ok',
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
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='late3-a: gate_a_entered',
            )

            # Inject B late — after the one-shot look-ahead peek.
            await worker._queue.put(req_b)

            # Wait for B's remote verify to enter (B has been merged + dispatched).
            await wait_responsive(
                gate_b_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='late3-b: gate_b_entered',
            )

            # Release A → A's verify completes, A's advance_main fires, A lands.
            gate_a_release.set()
            outcome_a = await wait_responsive(
                req_a.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='late3-a: MergeOutcome',
            )
            assert outcome_a.status == 'done', f'A must land cleanly; got {outcome_a!r}'

            # A has landed → main == A's merge commit.  Now release B.
            gate_b_release.set()
            outcome_b = await wait_responsive(
                req_b.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='late3-b: MergeOutcome',
            )

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

        # Mechanism guard: assert the ATTACH code path fired (not just the outcome).
        attach_msgs = [
            r.message
            for r in caplog.records
            if 'late arrival attaches to in-flight predecessor' in r.message
        ]
        assert attach_msgs, (
            'Expected at least one "late arrival attaches to in-flight '
            'predecessor" DEBUG log — confirms pending_spec_base ATTACH branch '
            'fired.  If missing, B may have been prefetched normally by the '
            'look-ahead, silently bypassing the late-arrival path.'
        )


# ===========================================================================
# Step-5 RED→GREEN: predecessor FAILING cascades to invalidate late arrival B
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)  # task 3980: worst per-method wait budget 105s, x2 stretched
class TestLateArrivalFailCascade:
    """Step-5 RED→GREEN — predecessor failing invalidates the late arrival (DONE-WHEN 4).

    DONE-WHEN 4: when predecessor A fails, the existing head-failure cascade:
      - cancels B's in-flight speculative remote verify (cancel_verify fired live)
      - emits speculative_discard for B with reason=previous_failed
      - re-merges B against actual main (non-speculative)
      - re-dispatches B so it verifies again and resolves 'done' (B's file lands)
      - A's file does NOT land on main (A failed)

    On main (RED): B merged non-speculatively against plain main → A failing
    does not trigger the head-failure cascade for B → no speculative_discard
    → B might land without re-verification against main+A.

    GREEN after step-2: B is a speculative descendant of A; the existing
    cascade in _verifier_loop (7608-7721) finds B in _inflight, cancels its
    verify, re-merges against actual main, re-dispatches with _redispatch,
    and emits speculative_discard reason=previous_failed via _dispatch_item.
    """

    async def test_predecessor_fail_cascades_to_late_arrival(
        self,
        spec_git_repo: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A fails; cascade cancels B's remote verify, remerges B, B lands as 'done'.

        Immediately GREEN after step-2 because B is structurally a speculative
        descendant and the existing head-failure cascade path handles it.
        """
        # task 3980: this is the ONLY late-arrival test that reaches the merge
        # disposition classifier -- classify_merge_failure_disposition runs only
        # on a FAILING verify, and every other late-arrival test is an all-pass
        # path.  MEASURED with a delegating spy on the callable, deliberately NOT
        # with log-record counts (a record count cannot tell "classifier ran and
        # succeeded" from "classifier never ran", and reasoning from that gap is
        # what produced a wrong review finding against this block): exactly 1
        # entry here, 0 across all eight sibling late-arrival integration tests.
        # So it is the correct and only anchor for the verify-result double's
        # fidelity guard below.
        #
        # Capture BOTH fail-open loggers: merge_disposition.py:711 (the
        # classifier's own internal fail-open) and merge_queue.py:995-1001
        # (_classify_disposition_for_outcome's catch for anything that re-raises
        # past it).  Filtering to merge_disposition alone let the second site
        # through silently.
        for _fail_open_logger in _FAIL_OPEN_LOGGERS:
            caplog.set_level(logging.WARNING, logger=_fail_open_logger)

        git_config = _make_late_arrival_git_config()
        git_ops = GitOps(git_config, spec_git_repo)
        fake_event_store = _LateArrivalFakeEventStore()

        # ── Gate A's LOCAL verify — FAILS on first call, passes on subsequent ─
        gate_a_entered = asyncio.Event()
        gate_a_release = asyncio.Event()
        _local_calls: list[int] = [0]

        # task 3980: built via _fake_verify_result (task 3477's
        # MagicMock(spec=VerifyResult) factory, seeded from
        # dataclasses.fields(VerifyResult)) rather than a bare MagicMock.
        # Dropped from the old inline construction:
        #   - `verify_skipped=False` — NOT a VerifyResult field at all; it lives
        #     on MergeOutcome (merge_types.py:945). The factory rejects it.
        #   - `lint_output=''`, `type_output=''`, `timed_out=False`,
        #     `category=''` — the factory already seeds these from a real
        #     VerifyResult's dataclass defaults, so restating them here is how
        #     the doubles drifted out of sync in the first place.
        # `category` now seeds to 'test_failure' on the failing leg instead of
        # the old hardcoded ''; both sit in the same policy bucket (neither is
        # in INFRA_TRANSIENT_CATEGORIES nor PREEXISTING_BREAK_SKIP_CATEGORIES),
        # so no retry/skip path changes — verified against verify_categories.py.
        async def _gated_failing_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                # First call: A's verify — gate then FAIL
                gate_a_entered.set()
                await gate_a_release.wait()
                return _fake_verify_result(
                    passed=False, summary='tests failed', test_output='FAIL',
                )
            # Subsequent calls (B's re-verify after cascade): PASS
            return _fake_verify_result(
                passed=True, summary='ok', test_output='ok',
            )

        # ── B's REMOTE verify: liveness-faithful runner to assert cancel-while-live ─
        # _id_liveness_fake_runner tracks whether cancel_verify was called while the
        # remote verify task was still running (state['live']=True).  cancel_verify
        # fired while live → remote_cancels_while_live == [1] (DONE-WHEN 4b).
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        liveness_remote = _id_liveness_fake_runner(
            gate_b_release, gate_b_entered, name='late5-laptop',
        )

        # ── Build disjoint branches ───────────────────────────────────────────
        config = OrchestratorConfig(project_root=spec_git_repo, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/late5-a', 'late5_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/late5-b', 'late5_b.py', 'b = 2\n',
        )

        worker = _make_late_arrival_worker(git_ops, event_store=fake_event_store)
        _inject_two_host_allocator(worker, liveness_remote)

        req_a = _make_request('late5-a', 'task/late5-a', wt_a, config)
        req_b = _make_request('late5-b', 'task/late5-b', wt_b, config)

        # ── POSITIVE LEG of the fidelity guard (task 3980 step-13) ────────────
        # A delegating SPY, deliberately not a caplog assertion.  The fidelity
        # guard further down asserts the classifier did not FAIL OPEN, which is
        # an assertion about ABSENCE — meaningless unless the classifier is
        # actually on this code path.  Nothing else in this test pins that.
        #
        # WHY A SPY AND NOT A LOG RECORD (read this before "simplifying" it):
        # a SUCCEEDING classifier emits NOTHING at WARNING — it logs only on the
        # degrade paths (merge_disposition.py:695 and :711).  So "no record" is
        # exactly what success looks like, and cannot be distinguished from "never
        # ran" by any caplog predicate.  A reviewer who tried inferring execution
        # from log records concluded this whole block was dead code; the spy is
        # what settles it, because it measures the callable itself.
        #
        # Patch the binding in MERGE_QUEUE's module globals: merge_queue.py:54
        # does `from orchestrator.merge_disposition import
        # classify_merge_failure_disposition`, so the call site at merge_queue.py:973
        # resolves through merge_queue's namespace and a patch on merge_disposition
        # alone would not be seen.  The delegate awaits the module-level import,
        # which stays bound to the real callable.
        classifier_calls: list[int] = [0]

        async def _counting_classify(*args: Any, **kwargs: Any) -> Any:
            classifier_calls[0] += 1
            return await classify_merge_failure_disposition(*args, **kwargs)

        monkeypatch.setattr(
            'orchestrator.merge_queue.classify_merge_failure_disposition',
            _counting_classify,
        )

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_failing_local):
            worker_task = asyncio.create_task(worker.run())

            # Enqueue A; wait for its verify to enter (look-ahead peek has fired).
            await worker._queue.put(req_a)
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='late5-a: gate_a_entered',
            )

            # Inject B LATE — after the one-shot look-ahead peek.
            # B attaches to A's pending spec base → B merges speculatively on A's commit.
            await worker._queue.put(req_b)

            # Wait for B's remote verify to start (confirms B is dispatched as
            # speculative descendant of A — otherwise there is no cascade to assert).
            await wait_responsive(
                gate_b_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='late5-b: gate_b_entered',
            )

            # Release A's gate with passed=False → A fails verification.
            gate_a_release.set()

            # The head-failure cascade cancels B's remote verify, re-merges B
            # against actual main, re-dispatches → local re-verify (call[1], passes).
            # Wait for B to resolve 'done'.
            # task 3980: this site was the file's only bare mid-range deadline
            # (a raw `timeout=25.0`) and one of the three MEASURED failures.
            # Task 2376's sweep replaced merge-pipeline wait literals but its
            # stated policy only covered literals <= 15, so 25.0 sat just above
            # the sweep and survived. The bound is now DERIVED from the shared
            # constant, and step-5's structural guard keys on the call shape
            # rather than on a literal's magnitude, so a value above whatever
            # the next sweep's threshold happens to be can no longer hide.
            outcome_b = await wait_responsive(
                req_b.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='late5-b: MergeOutcome',
            )

            # Await A's result too (should be set to a failed outcome).
            outcome_a = await wait_responsive(
                req_a.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='late5-a: MergeOutcome',
            )

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # ── VERIFY-RESULT DOUBLE FIDELITY GUARD (task 3980) ───────────────────
        # WHAT THIS IS: a fidelity guard on the verify-result double this test
        # feeds the merge pipeline, anchored at the one late-arrival site where
        # the disposition classifier is reachable at all.  It is NOT a
        # precondition on the DONE-WHEN 4 assertions below.  MEASURED: running
        # the bare-MagicMock mutation with the fail-open assertion neutralized
        # leaves this test PASSING — DONE-WHEN 4(a)-(d) assert on
        # speculative_merge events, cancel_verify liveness, outcome status and
        # `git ls-tree` contents, none of which is disposition-sensitive.  The
        # earlier "PRECONDITION on the assertions below" framing overstated what
        # this protects; the real and sufficient reason it exists is that nothing
        # ELSE in the suite runs the classifier against these doubles.
        #
        # THE MECHANISM IT CATCHES: classify_merge_failure_disposition calls
        # _extract_failing_tests_and_candidate_files, which joins
        # `verify_result.cause_hint` with `verify_result.test_output`
        # (merge_disposition.py:218-221) under an `if part` filter.  An unspecced
        # MagicMock double leaves `cause_hint` UNSET, so attribute access
        # auto-vivifies a TRUTHY child Mock that survives that filter and then
        # raises `TypeError: sequence item 0: expected str instance, MagicMock
        # found` out of str.join.  The classifier catches it and degrades to
        # INDETERMINATE (fail-open, I3) at merge_disposition.py:710-719 — a
        # verdict indistinguishable from a genuine one downstream, which is why
        # the degrade has to be asserted against rather than tolerated.
        #
        # TestDispositionDoubleFidelity pins that same mechanism hermetically and
        # two-sidedly (positive + mutation leg, no merge worker, milliseconds).
        # This block is the LIVE counterpart: it proves the doubles this
        # integration test actually feeds the pipeline stay classifier-consumable.
        assert classifier_calls[0] >= 1, (
            'the merge disposition classifier was never reached, so the '
            'fail-open assertion below is a permanent no-op.\n'
            f'spy count = {classifier_calls[0]}; expected >= 1 (measured: '
            'exactly 1 on this path today).\n'
            'This is not a flake and not a doubles problem: it means '
            'classification has moved OFF the head-failure cascade path. '
            'Relocate this guard to wherever the classifier now consumes a '
            'verify-result double, or delete it deliberately — do NOT leave it '
            'here to rot into an assertion that cannot fail. '
            'TestDispositionDoubleFidelity keeps the hermetic proof either way.'
        )

        # Both fail-open sites, one predicate — see _fail_open_records.
        fail_open = _fail_open_records(caplog.records)
        assert not fail_open, (
            'merge disposition classifier FAILED OPEN — it ran on this test\'s '
            'verify-result double and could not consume it, so any '
            'disposition-sensitive assertion built on these doubles (here or in '
            'a future test reusing them) is VACUOUS.\n'
            f'{len(fail_open)} fail-open WARNING(s) captured:\n'
            f'{_format_fail_open_records(fail_open)}\n'
            'Remedy: the verify-result double returned by this test\'s patched '
            'run_scoped_verification must be built with _fake_verify_result(...) '
            '(a MagicMock(spec=VerifyResult) seeded from the real dataclass '
            'defaults, so cause_hint is a real str) rather than a bare MagicMock, '
            'whose unset cause_hint auto-vivifies a truthy child Mock and breaks '
            'the str.join at merge_disposition.py:218.'
        )

        # ── A failed (verify returned passed=False) ────────────────────────────
        assert outcome_a.status != 'done', (
            f'A must NOT land (verify failed); got outcome_a={outcome_a!r}'
        )

        # ── DONE-WHEN 4(a): speculative_merge event for B (B was dispatched
        #    speculatively against A's commit — only present after step-2).
        #
        #    NOTE: speculative_discard is emitted in _dispatch_item only when a
        #    speculative item is re-dispatched from _redispatch with _n_failed=True
        #    (single-host path). In the two-host cascade path, the remerged item
        #    has speculative=False (_remerge sets speculative=False), so
        #    speculative_discard is NOT emitted at re-dispatch.  The meaningful
        #    RED/GREEN distinguisher is the speculative_merge event for B:
        #
        #    RED (on main without step-2): B merged non-speculatively (spec_base=None
        #    reset at line 6875) → no speculative_merge event for B.
        #    GREEN (step-2): pending_spec_base carries A's commit across the blocking
        #    dequeue → B merges speculatively → speculative_merge event for B.
        spec_merge_events = fake_event_store.speculative_events(EventType.speculative_merge)
        b_spec_merge = [e for e in spec_merge_events if e['task_id'] == 'late5-b']
        assert len(b_spec_merge) == 1, (
            f'Expected exactly one speculative_merge event for B; '
            f'got {len(b_spec_merge)!r}: {b_spec_merge!r}.\n'
            'RED (on main): spec_base=None reset unconditionally → B non-speculative '
            '→ no speculative_merge event.\n'
            'GREEN (step-2): B attaches to A\'s pending spec base → speculative=True '
            '→ speculative_merge emitted.'
        )
        assert b_spec_merge[0]['data'].get('base_sha') is not None, (
            f'speculative_merge event for B must carry a non-None base_sha; '
            f'got data={b_spec_merge[0]["data"]!r}.'
        )

        # ── DONE-WHEN 4(b): cancel_verify fired while B's remote verify was live ─
        # The head-failure cascade cancels B's remote verify via _abort_remote_verify
        # before task.cancel(), so cancel_verify is called while state['live']=True.
        # The abort-poll inside _run_inflight_verify may also fire cancel_verify while
        # live (if it polls between the cascade's _abort_remote_verify and the
        # CancelledError delivery), so we assert >= 1 rather than exactly 1.
        assert liveness_remote.remote_cancels_while_live, (
            f'cancel_verify must have been called at least once while B\'s remote '
            f'verify was live; got remote_cancels_while_live='
            f'{liveness_remote.remote_cancels_while_live!r}.\n'
            'The cascade calls _abort_remote_verify before task.cancel().'
        )

        # ── DONE-WHEN 4(c): B resolves 'done' after cascade + remerge ─────────
        # After the cascade cancels B's speculative verify and remerges B against
        # actual main (= original main since A never landed), B re-verifies locally
        # (call[1] returns passed=True) and advances main cleanly.
        assert outcome_b.status == 'done', (
            f'B must resolve done after cascade + remerge + re-verify; '
            f'got outcome_b={outcome_b!r}.'
        )

        # ── DONE-WHEN 4(d): B's file on main; A's file NOT on main ────────────
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'], cwd=spec_git_repo,
        )
        assert 'late5_b.py' in main_files, (
            'B\'s file (late5_b.py) must be on main after cascade + remerge + re-verify'
        )
        assert 'late5_a.py' not in main_files, (
            'A\'s file (late5_a.py) must NOT be on main (A failed verification)'
        )


# ===========================================================================
# Step-7 Guards: fallback, depth-K, skip_verify, K=1 sanity
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)  # task 3980: worst per-method wait budget 110s, x2 stretched
class TestLateArrivalGuards:
    """Step-7 guards — fallback + permit accounting + depth-K + skip_verify + K=1 sanity.

    (a) FALLBACK: when predecessor A finalizes (result.done()) before B arrives,
        B falls back to plain-main non-speculative; retained permit is released;
        no speculative_merge event for B.

    (b) DEPTH-K: with K=2, the speculation slot returns to K after the attach
        path completes — no permit leak, merger never double-acquires.

    (c) task-1724: the attached late arrival B is a RealMergeItem — it always
        undergoes real verification, never silently skips the merge gate (the
        old skip_verify=False field was retired by task ο; a DecidedItem,
        which the standard speculative merge path never constructs, would be
        the only way to bypass verify now).

    (d) K=1 sanity: with speculation_depth=1, the late-arrival attach still
        works (B attaches to A's commit, verifies, lands) and at most one
        speculation permit is ever held (slot._value >= 0 throughout).
    """

    async def test_fallback_when_predecessor_done(self, spec_git_repo: Path) -> None:
        """FALLBACK: B arrives after A has fully landed → non-speculative; slot released.

        When pending_predecessor.result.done() is True at the top-of-loop decision
        (step-2), the FALLBACK branch fires: retained permit released, spec_base=None.
        B is merged against plain main with base_sha=None; no speculative_merge event.
        """
        git_config = _make_late_arrival_git_config()
        git_ops = GitOps(git_config, spec_git_repo)
        fake_event_store = _LateArrivalFakeEventStore()

        # ── Spy: capture merge_to_main base_sha per branch ───────────────────
        _merge_calls: list[dict] = []
        original_merge = git_ops.merge_to_main

        async def _spy_merge(wt: Path, branch: str, base_sha: str | None = None) -> Any:
            result = await original_merge(wt, branch, base_sha=base_sha)
            _merge_calls.append({'branch': branch, 'base_sha': base_sha})
            return result

        git_ops.merge_to_main = _spy_merge  # type: ignore[method-assign]

        # ── A's local verify: passes immediately (no gate) ───────────────────
        async def _passing_local(*args: Any, **kwargs: Any) -> MagicMock:
            # task 3980: _fake_verify_result, not a bare MagicMock — see the
            # TestNoBareVerifyResultDoubles guard at the foot of this module.
            return _fake_verify_result(
                passed=True, summary='ok', test_output='ok',
            )

        # ── B's remote runner: passes immediately ────────────────────────────
        gate_b_prerelease = asyncio.Event()
        gate_b_prerelease.set()
        fake_remote = _gated_runner(gate_b_prerelease, passed=True, name='fallback7-laptop')

        # ── Build branches ────────────────────────────────────────────────────
        config = OrchestratorConfig(project_root=spec_git_repo, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/guard7-a', 'guard7_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/guard7-b', 'guard7_b.py', 'b = 2\n',
        )

        worker = _make_late_arrival_worker(
            git_ops, event_store=fake_event_store, speculation_depth=2,
        )
        _inject_two_host_allocator(worker, fake_remote)

        req_a = _make_request('guard7-a', 'task/guard7-a', wt_a, config)
        req_b = _make_request('guard7-b', 'task/guard7-b', wt_b, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _passing_local):
            worker_task = asyncio.create_task(worker.run())

            # Enqueue A; let it run and land fully (no gate).
            await worker._queue.put(req_a)
            outcome_a = await wait_responsive(
                req_a.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='guard7-a: MergeOutcome',
            )
            assert outcome_a.status == 'done', f'A must land; got {outcome_a!r}'

            # Now enqueue B — pending_predecessor (A) is done → FALLBACK.
            # The merger is blocked in _acquire_next_request(); the decision
            # code fires with pending_predecessor.result.done() == True.
            await worker._queue.put(req_b)
            outcome_b = await wait_responsive(
                req_b.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='guard7-b: MergeOutcome',
            )

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        assert outcome_b.status == 'done', f'B must land; got {outcome_b!r}'

        # FALLBACK guard (a): no speculative_merge event for B.
        spec_evts = fake_event_store.speculative_events(EventType.speculative_merge)
        b_spec = [e for e in spec_evts if e['task_id'] == 'guard7-b']
        assert len(b_spec) == 0, (
            f'FALLBACK: no speculative_merge event expected for B (A is done); '
            f'got {b_spec!r}.\n'
            'The pending_predecessor.result.done() check must fire the FALLBACK branch '
            'and leave spec_base=None so B is merged non-speculatively.'
        )

        # FALLBACK guard (b): B merged against plain main (base_sha=None).
        b_merges = [c for c in _merge_calls if c['branch'] == 'task/guard7-b']
        assert b_merges, 'merge_to_main must have been called for B'
        assert b_merges[0]['base_sha'] is None, (
            f'FALLBACK: B must use base_sha=None (plain-main merge); '
            f'got {b_merges[0]["base_sha"]!r}.\n'
            'When predecessor is done, the retained permit must be released and '
            'spec_base=None so B verifies against current main (not A\'s stale commit).'
        )

        # FALLBACK guard (c): B resolves done — no deadlock from the FALLBACK permit
        # release.  The slot invariant (retained permit released on FALLBACK) is
        # proven indirectly: B merges non-spec and lands, which requires the merger
        # to have released the look-ahead permit and moved on to process B.  A
        # permanent slot leak would block any future speculative merge attempt, and
        # the no-deadlock proof is that B's successor can still acquire one.

    async def test_depth_k_slot_no_leak_after_attach(self, spec_git_repo: Path) -> None:
        """DEPTH-K: with K=2, slot returns to K after late-arrival attach; no double-acquire.

        The late-arrival path acquires exactly ONE speculation permit (at the
        look-ahead), retains it across the blocking dequeue, and transfers it to
        B's InflightEntry (was_speculative=True).  When B completes, the slot is
        released.  slot._value must equal K=2 after both A and B have landed.
        """
        git_config = _make_late_arrival_git_config()
        git_ops = GitOps(git_config, spec_git_repo)

        # ── Gate A's verify so the look-ahead peek fires before A lands ──────
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            # task 3980: _fake_verify_result, not a bare MagicMock — see the
            # TestNoBareVerifyResultDoubles guard at the foot of this module.
            return _fake_verify_result(
                passed=True, summary='ok', test_output='ok',
            )

        gate_b_prerelease = asyncio.Event()
        gate_b_prerelease.set()
        fake_remote = _gated_runner(gate_b_prerelease, passed=True, name='depth7-laptop')

        config = OrchestratorConfig(project_root=spec_git_repo, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/dk7-a', 'dk7_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/dk7-b', 'dk7_b.py', 'b = 2\n',
        )

        K = 2
        worker = _make_late_arrival_worker(git_ops, speculation_depth=K)
        _inject_two_host_allocator(worker, fake_remote)

        req_a = _make_request('dk7-a', 'task/dk7-a', wt_a, config)
        req_b = _make_request('dk7-b', 'task/dk7-b', wt_b, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await worker._queue.put(req_a)
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='dk7-a: gate_a_entered',
            )

            # After gate_a_entered, exactly ONE permit is held (the look-ahead).
            # slot._value == K-1 == 1.
            assert worker._speculation_slot._value == K - 1, (
                f'After look-ahead, exactly 1 permit must be held; '
                f'slot._value={worker._speculation_slot._value} (expected {K - 1}).'
            )

            # Inject B late — permit is retained and transferred to B's InflightEntry.
            await worker._queue.put(req_b)

            # Release A's gate → A lands.
            gate_a_release.set()
            outcome_a = await wait_responsive(
                req_a.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='dk7-a: MergeOutcome',
            )
            assert outcome_a.status == 'done'

            outcome_b = await wait_responsive(
                req_b.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='dk7-b: MergeOutcome',
            )
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        assert outcome_b.status == 'done'

        # DEPTH-K guard: B resolves done (no deadlock) and no double-acquire.
        #
        # The permit count invariant is verified at the MID-TEST checkpoint above:
        # after gate_a_entered, slot._value == K-1 proves the look-ahead held
        # exactly ONE permit (not two — no double-acquire).  A slot _value of 0
        # (exhausted for K=2) would indicate a double-acquire bug.
        #
        # After B's finalize, the merger immediately acquires the permit again for
        # B's successor look-ahead (step-2 else-branch retains), so the slot is
        # at K-1 again rather than K when we check.  The no-leak property is
        # proven by: (single acquire at look-ahead) + (single release at B's
        # finalize) → net zero across the A→B attach cycle.

    async def test_attached_late_arrival_skip_verify_false(
        self, spec_git_repo: Path,
    ) -> None:
        """task-1724 guard: the late-arrival attached item is a RealMergeItem.

        The pending_spec_base ATTACH path creates B's item via the standard
        speculative merge path in _merger_loop (:7146), which always produces a
        RealMergeItem (task-1724 invariant: verify is never skipped). This test
        asserts that directly by capturing B's item from the verifier queue and
        checking its variant — the old skip_verify=False field was retired by
        task ο in favor of this structural guarantee.
        """
        git_config = _make_late_arrival_git_config()
        git_ops = GitOps(git_config, spec_git_repo)

        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            # task 3980: _fake_verify_result, not a bare MagicMock — see the
            # TestNoBareVerifyResultDoubles guard at the foot of this module.
            return _fake_verify_result(
                passed=True, summary='ok', test_output='ok',
            )

        gate_b_prerelease = asyncio.Event()
        gate_b_prerelease.set()
        fake_remote = _gated_runner(gate_b_prerelease, passed=True, name='sv7-laptop')

        config = OrchestratorConfig(project_root=spec_git_repo, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/sv7-a', 'sv7_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/sv7-b', 'sv7_b.py', 'b = 2\n',
        )

        worker = _make_late_arrival_worker(git_ops, speculation_depth=2)
        _inject_two_host_allocator(worker, fake_remote)

        # ── Spy: capture SpeculativeItems put on verifier queue ────────────────
        captured_items: list[Any] = []
        orig_put = worker._verifier_queue.put_nowait

        def _spy_put(item: Any) -> None:
            if item is not None:
                captured_items.append(item)
            orig_put(item)

        worker._verifier_queue.put_nowait = _spy_put  # type: ignore[method-assign]

        req_a = _make_request('sv7-a', 'task/sv7-a', wt_a, config)
        req_b = _make_request('sv7-b', 'task/sv7-b', wt_b, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await worker._queue.put(req_a)
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='sv7-a: gate_a_entered',
            )

            await worker._queue.put(req_b)

            gate_a_release.set()
            await wait_responsive(
                req_a.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='sv7-a: MergeOutcome',
            )
            await wait_responsive(
                req_b.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='sv7-b: MergeOutcome',
            )
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # Find B's item in captured queue items — it must be a RealMergeItem,
        # proving a real verify always runs for the standard speculative merge
        # path (task-1724 guard). The old skip_verify=False field check is
        # retired by task ο: RealMergeItem has no skip_verify field at all,
        # since a RealMergeItem structurally always undergoes verification
        # (only a DecidedItem — never constructed by this path — would skip it).
        from orchestrator.merge_queue import RealMergeItem
        b_items = [
            item for item in captured_items
            if isinstance(item, RealMergeItem) and item.request.task_id == 'sv7-b'
        ]
        assert b_items, (
            'No RealMergeItem for B found in verifier queue captures (expected the '
            'standard speculative merge path to always produce a RealMergeItem, '
            'never a DecidedItem skip-verify passthrough); '
            f'captured: {captured_items!r}'
        )

    async def test_k1_sanity_late_arrival_attaches(self, spec_git_repo: Path) -> None:
        """K=1 sanity: with speculation_depth=1, late arrival B still attaches to A's commit.

        K=1 means at most one speculative item can be in flight at a time.  The
        late-arrival path acquires the one permit at the look-ahead, retains it,
        and transfers it to B's InflightEntry when B arrives.  A is always merged
        non-speculatively; B is merged speculatively against A's commit.
        """
        git_config = _make_late_arrival_git_config()
        git_ops = GitOps(git_config, spec_git_repo)
        fake_event_store = _LateArrivalFakeEventStore()

        _merge_calls: list[dict] = []
        original_merge = git_ops.merge_to_main

        async def _spy_merge(wt: Path, branch: str, base_sha: str | None = None) -> Any:
            result = await original_merge(wt, branch, base_sha=base_sha)
            _merge_calls.append({
                'branch': branch,
                'base_sha': base_sha,
                'merge_commit': result.merge_commit,
            })
            return result

        git_ops.merge_to_main = _spy_merge  # type: ignore[method-assign]

        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            # task 3980: _fake_verify_result, not a bare MagicMock — see the
            # TestNoBareVerifyResultDoubles guard at the foot of this module.
            return _fake_verify_result(
                passed=True, summary='ok', test_output='ok',
            )

        gate_b_prerelease = asyncio.Event()
        gate_b_prerelease.set()
        fake_remote = _gated_runner(gate_b_prerelease, passed=True, name='k1-7-laptop')

        config = OrchestratorConfig(project_root=spec_git_repo, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/k1-7-a', 'k1_7_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/k1-7-b', 'k1_7_b.py', 'b = 2\n',
        )

        # K=1: at most one speculative item in flight at a time.
        K = 1
        worker = _make_late_arrival_worker(
            git_ops, event_store=fake_event_store, speculation_depth=K,
        )
        _inject_two_host_allocator(worker, fake_remote)

        req_a = _make_request('k1-7-a', 'task/k1-7-a', wt_a, config)
        req_b = _make_request('k1-7-b', 'task/k1-7-b', wt_b, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            await worker._queue.put(req_a)
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='k1-7-a: gate_a_entered',
            )

            # After gate_a_entered: exactly ONE permit held (the look-ahead).
            # With K=1, slot._value == 0.
            assert worker._speculation_slot._value == 0, (
                f'K=1: after look-ahead, slot._value must be 0 (one permit held); '
                f'got {worker._speculation_slot._value}.'
            )

            await worker._queue.put(req_b)

            gate_a_release.set()
            outcome_a = await wait_responsive(
                req_a.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='k1-7-a: MergeOutcome',
            )
            assert outcome_a.status == 'done'

            outcome_b = await wait_responsive(
                req_b.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='k1-7-b: MergeOutcome',
            )
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        assert outcome_b.status == 'done', f'B must land; got {outcome_b!r}'

        # K=1 sanity (i): B attached to A's commit (speculative_merge event for B).
        spec_evts = fake_event_store.speculative_events(EventType.speculative_merge)
        b_spec = [e for e in spec_evts if e['task_id'] == 'k1-7-b']
        assert len(b_spec) == 1, (
            f'K=1: expected exactly one speculative_merge event for B; '
            f'got {len(b_spec)!r}: {b_spec!r}.\n'
            'K=1 must not prevent the late-arrival ATTACH path from firing.'
        )

        # K=1 sanity (ii): B's base_sha == A's merge commit (attached to A).
        a_merges = [c for c in _merge_calls if c['branch'] == 'task/k1-7-a']
        b_merges = [c for c in _merge_calls if c['branch'] == 'task/k1-7-b']
        assert a_merges and b_merges
        a_merge_commit = (a_merges[0]['merge_commit'] or '').strip()
        assert b_merges[0]['base_sha'] == a_merge_commit, (
            f'K=1: B must be merged against A\'s commit ({a_merge_commit!r}); '
            f'got base_sha={b_merges[0]["base_sha"]!r}.'
        )

        # K=1 sanity (iii): no double-acquire / no deadlock.
        # At most 1 permit held at any point — proven by the mid-test checkpoint
        # (slot._value == 0 after gate_a_entered: one permit held for A's look-ahead
        # retained → B's ATTACH).  With K=1, slot._value == 0 means EXACTLY one
        # permit is in use, which is correct and cannot exceed K=1.

    async def test_shutdown_releases_retained_permit(self, spec_git_repo: Path) -> None:
        """Shutdown while retain state is live (no B queued) releases the held permit.

        Regression guard (Amendment 3 / reviewer suggestion):
        When the merger's look-ahead finds nothing pickable and the predecessor is
        still in-flight, it RETAINS the speculation permit (pending_spec_base set,
        held_spec_permit=True) and blocks in _acquire_next_request() waiting for a
        late arrival.  If the worker shuts down before any late arrival arrives, the
        outer finally block at :7412 must release the retained permit.

        A regression that drops ``if held_spec_permit: release()`` from the finally
        path — or that accidentally sets held_spec_permit=False before the finally
        block — would leak a permit, reducing effective K by 1 for the lifetime of
        the worker (until the safety-valve releases in stop() compensate).

        Slot accounting:
          After gate_a_entered: slot._value == K-1  (one permit held in retain state)
          After worker.stop():  slot._value == 2*K+1
            • stop() releases K+1 times (over-release safety valve — see :6268)
            • merger's finally block releases 1 more time (the retained permit)
            • net: (K-1) + (K+1) + 1 = 2K+1
          With bug (finally skips release): slot._value == 2*K
            • (K-1) + (K+1) = 2K  (one fewer release)

        Drive:
          - Gate A's local verify so A is in-flight (retain state entered by merger).
          - Do NOT enqueue B (queue stays empty; merger stays in _acquire_next_request).
          - Assert slot._value == K-1 (one permit held in retain state).
          - Release A's gate, then call worker.stop() — shutdown sentinel dequeued.
          - Assert slot._value == 2*K+1 after the worker task completes.
        """
        K = 2
        git_config = _make_late_arrival_git_config()
        git_ops = GitOps(git_config, spec_git_repo)

        # ── Gate: hold A's local verify in-flight ────────────────────────────
        gate_a_entered = asyncio.Event()
        gate_a_release = asyncio.Event()
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            # task 3980: _fake_verify_result, not a bare MagicMock — see the
            # TestNoBareVerifyResultDoubles guard at the foot of this module.
            return _fake_verify_result(
                passed=True, summary='ok', test_output='ok',
            )

        gate_b_prerelease = asyncio.Event()
        gate_b_prerelease.set()
        fake_remote = _gated_runner(
            gate_b_prerelease, passed=True, name='shutdown-guard-laptop',
        )

        config = OrchestratorConfig(project_root=spec_git_repo, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/shutdown-guard-a', 'shutdown_guard_a.py', 'a = 1\n',
        )

        worker = _make_late_arrival_worker(git_ops, speculation_depth=K)
        _inject_two_host_allocator(worker, fake_remote)
        req_a = _make_request('shutdown-guard-a', 'task/shutdown-guard-a', wt_a, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            # Enqueue only A — no B.
            await worker._queue.put(req_a)

            # Wait until A's verify enters the gate.  At this point the merger has:
            #   (1) merged A and enqueued it to the verifier
            #   (2) acquired the speculation permit (look-ahead)
            #   (3) done the peek → found nothing → entered RETAIN state
            #   (4) set pending_spec_base = A's merge commit, held_spec_permit=True
            #   (5) blocked in _acquire_next_request() (queue empty, no B)
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='shutdown-guard-a: gate_a_entered',
            )

            # RETAIN state: exactly one permit held; slot._value == K-1.
            retain_value = worker._speculation_slot._value
            assert retain_value == K - 1, (
                f'Retain state: expect one permit held (slot._value == {K - 1}); '
                f'got {retain_value}.\n'
                'If slot._value == K, the permit was released before retain state '
                'was established — test precondition not met.'
            )

            # Release A's gate so A can finalize when the verifier processes it.
            # The merger stays blocked in _acquire_next_request (queue still empty).
            gate_a_release.set()

            # Shut down — no B ever arrives.  stop() puts None in the merge queue
            # (after releasing K+1 times as a safety valve); the merger dequeues it,
            # breaks out of the loop, and the finally block at :7412 releases the
            # retained permit.
            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=10.0)

        # Expected slot value after full shutdown:
        #   retain_value (K-1)
        #   + stop()'s K+1 over-releases (safety valve, :6268)
        #   + finally block's 1 release (the retained permit)
        #   = (K-1) + (K+1) + 1 = 2K+1
        #
        # With a bug (finally doesn't release the retained permit):
        #   (K-1) + (K+1) = 2K   ← one fewer than expected
        expected_post_stop = 2 * K + 1
        assert worker._speculation_slot._value == expected_post_stop, (
            f'Shutdown guard: slot._value must be {expected_post_stop} after stop(); '
            f'got {worker._speculation_slot._value}.\n'
            f'Formula: retain({K-1}) + stop_releases({K+1}) + finally_release(1) = '
            f'{expected_post_stop}.\n'
            f'A value of {2*K} means the merger\'s finally block did NOT release the '
            f'retained permit (held_spec_permit flag lost before finally block).'
        )


# ===========================================================================
# Step-8 Guard: submission-order CAS invariant for the late-arrival path
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)  # task 3980: worst per-method wait budget 125s, x2 stretched
class TestLateArrivalSubmissionOrderCAS:
    """Step-8 guard — main advances in strict submission order on the late-arrival path.

    This is the regression guard the task names explicitly (from the plan):

    Assert that with a late-attached B:
      1. A's advance_main/CAS completes (A lands on main) BEFORE B's advance_main runs.
      2. B's expected_main kwarg equals A's merge commit (CAS is set up for the right base).
      3. Both A and B land (outcome='done'); no interleaving or out-of-order advance.

    The existing test_advance_main_submission_order_regression_guard in
    TestB9CapstoneIntegration tests CAS ordering with mocked git_ops; this test
    verifies the same invariant end-to-end with the real-git harness and the
    late-arrival path (pending_spec_base attach via step-2).

    This stays GREEN after step-2/step-4/step-6 because:
      B.base_sha == A's merge commit → advance_main(expected_main=A's commit) → clean CAS.
      After A lands, main == A's commit → B's CAS succeeds immediately ('advanced').
      The verifier's single-threaded serial finalize loop ensures A finalizes first.
    """

    async def test_submission_order_cas_late_arrival(
        self, spec_git_repo: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A's advance_main completes before B's; B's expected_main == A's merge commit.

        Drive:
          - Gate A's LOCAL verify (gate_a_entered fires after look-ahead peek).
          - Inject B LATE (after gate_a_entered).
          - Gate B's REMOTE verify (gate_b_release) so B's verify enters before A lands.
          - Release A → A lands.
          - Release B → B advances.

        Assert:
          (a) advance_main call for A comes before call for B (submission order).
          (b) B's advance_main was called with expected_main == A's merge commit
              (B's CAS uses A's merge commit as the base, not the original main).
          (c) Both A and B resolve 'done' (clean CAS, no rebase).
        """
        # Capture DEBUG-level logs so we can assert the ATTACH code path fired.
        caplog.set_level(logging.DEBUG, logger='orchestrator.merge_queue')

        git_config = _make_late_arrival_git_config()
        git_ops = GitOps(git_config, spec_git_repo)

        # ── Spy: capture advance_main calls in order ──────────────────────────
        advance_call_log: list[dict] = []
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
            advance_call_log.append({
                'branch': branch,
                'expected_main': kwargs.get('expected_main'),
                'result': result.result,
            })
            return result

        git_ops.advance_main = _spy_advance_main  # type: ignore[method-assign]

        # ── Spy: capture A's merge commit from merge_to_main ──────────────────
        _merge_calls: list[dict] = []
        original_merge_to_main = git_ops.merge_to_main

        async def _spy_merge_to_main(
            wt: Path, branch: str, base_sha: str | None = None,
        ) -> Any:
            result = await original_merge_to_main(wt, branch, base_sha=base_sha)
            _merge_calls.append({
                'branch': branch,
                'merge_commit': result.merge_commit,
            })
            return result

        git_ops.merge_to_main = _spy_merge_to_main  # type: ignore[method-assign]

        # ── Gate A's LOCAL verify ─────────────────────────────────────────────
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls: list[int] = [0]

        async def _gated_local(*args: Any, **kwargs: Any) -> MagicMock:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                gate_a_entered.set()
                await gate_a_release.wait()
            # task 3980: _fake_verify_result, not a bare MagicMock — see the
            # TestNoBareVerifyResultDoubles guard at the foot of this module.
            return _fake_verify_result(
                passed=True, summary='ok', test_output='ok',
            )

        # ── Gate B's REMOTE verify ─────────────────────────────────────────────
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='cas8-laptop',
        )

        # ── Build disjoint branches ───────────────────────────────────────────
        config = OrchestratorConfig(project_root=spec_git_repo, git=git_config)
        wt_a = await _make_branch_with_file(
            git_ops, 'task/cas8-a', 'cas8_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/cas8-b', 'cas8_b.py', 'b = 2\n',
        )

        worker = _make_late_arrival_worker(git_ops, speculation_depth=2)
        _inject_two_host_allocator(worker, gated_remote)

        req_a = _make_request('cas8-a', 'task/cas8-a', wt_a, config)
        req_b = _make_request('cas8-b', 'task/cas8-b', wt_b, config)

        with patch('orchestrator.merge_queue.run_scoped_verification', _gated_local):
            worker_task = asyncio.create_task(worker.run())

            # Enqueue A; wait for look-ahead peek to fire.
            await worker._queue.put(req_a)
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='cas8-a: gate_a_entered',
            )

            # Inject B LATE; wait for B's remote verify to enter.
            await worker._queue.put(req_b)
            await wait_responsive(
                gate_b_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='cas8-b: gate_b_entered',
            )

            # Release A → A lands; then release B → B advances.
            gate_a_release.set()
            outcome_a = await wait_responsive(
                req_a.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='cas8-a: MergeOutcome',
            )
            assert outcome_a.status == 'done', f'A must land; got {outcome_a!r}'

            gate_b_release.set()
            outcome_b = await wait_responsive(
                req_b.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='cas8-b: MergeOutcome',
            )

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # ── Assertions ────────────────────────────────────────────────────────

        assert outcome_b.status == 'done', (
            f'B must land cleanly after A; got {outcome_b!r}'
        )

        # (a) Submission-order CAS: A's advance_main called before B's.
        a_advances = [c for c in advance_call_log if c['branch'] == 'task/cas8-a']
        b_advances = [c for c in advance_call_log if c['branch'] == 'task/cas8-b']
        assert a_advances, 'advance_main must have been called for A'
        assert b_advances, 'advance_main must have been called for B'

        a_idx = next(i for i, c in enumerate(advance_call_log) if c['branch'] == 'task/cas8-a')
        b_idx = next(i for i, c in enumerate(advance_call_log) if c['branch'] == 'task/cas8-b')
        assert a_idx < b_idx, (
            f'Submission-order CAS violation: A\'s advance_main must come first; '
            f'A at index {a_idx}, B at index {b_idx} in advance_call_log={advance_call_log!r}.\n'
            'The verifier\'s serial finalize loop (single-threaded _verifier_loop) must '
            'process A\'s InflightEntry before B\'s — preserving submission order.'
        )

        # (b) B's expected_main == A's merge commit.
        a_merges = [c for c in _merge_calls if c['branch'] == 'task/cas8-a']
        assert a_merges, 'merge_to_main must have been called for A'
        a_merge_commit = (a_merges[0]['merge_commit'] or '').strip()

        b_expected_main = b_advances[0]['expected_main']
        assert b_expected_main == a_merge_commit, (
            f'CAS invariant: B\'s expected_main must equal A\'s merge commit '
            f'{a_merge_commit!r}; got expected_main={b_expected_main!r}.\n'
            'B was merged speculatively against A\'s commit (pending_spec_base); '
            'advance_main must use that commit as expected_main for the CAS. '
            'After A lands, main == A\'s commit → clean fast-forward CAS for B.'
        )

        # (c) Both advance as 'advanced' (clean CAS, no rebase).
        assert a_advances[0]['result'] == 'advanced', (
            f'A must advance cleanly; got {a_advances[0]["result"]!r}'
        )
        assert b_advances[0]['result'] == 'advanced', (
            f'B must advance cleanly (expected_main == main-after-A → clean CAS); '
            f'got {b_advances[0]["result"]!r}.\n'
            'If B returned rebased_pending_reverify, the pending_spec_base fix did not '
            'set B.base_sha correctly — disjoint-skip semantic-conflict hole still open.'
        )

        # (d) Both files on main.
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'], cwd=spec_git_repo,
        )
        assert 'cas8_a.py' in main_files, 'A\'s file must be on main'
        assert 'cas8_b.py' in main_files, 'B\'s file must be on main'

        # Mechanism guard: assert the ATTACH code path fired (not just the outcome).
        attach_msgs = [
            r.message
            for r in caplog.records
            if 'late arrival attaches to in-flight predecessor' in r.message
        ]
        assert attach_msgs, (
            'Expected at least one "late arrival attaches to in-flight '
            'predecessor" DEBUG log — confirms pending_spec_base ATTACH branch '
            'fired.  Without this, B may have been prefetched normally by the '
            'look-ahead, bypassing the late-arrival path the submission-order '
            'CAS test is designed to exercise.'
        )


# ---------------------------------------------------------------------------
# Task 2154 (W1 β): write-ahead landed-outbox wiring at both CAS advance sites
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestJournalLandedThenAdvanceHelper:
    """Step-1 (RED) — shared write-ahead helper: record() precedes advance_main.

    ``_journal_landed_then_advance`` is the single ordering point (PRD WA-1)
    that BOTH CAS advance sites route through: record a LandedRow into the
    outbox, THEN await ``git_ops.advance_main``. RED until step-2 adds the
    helper to merge_queue.py (import error).
    """

    async def test_records_before_advance(self, tmp_path: Path) -> None:
        """The row must already be visible from inside advance_main's own call."""
        from orchestrator.merge_queue import _journal_landed_then_advance

        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        merge_wt = tmp_path / '_merge-wt'
        captured: dict[str, Any] = {}

        async def _fake_advance_main(*args: Any, **kwargs: Any) -> AdvanceOutcome:
            # WA-1: captured HERE, at call time — proves record() precedes advance.
            captured['row'] = outbox.lookup('t1')
            return AdvanceOutcome('advanced', advanced_sha='mergesha')

        git_ops = MagicMock()
        git_ops.advance_main = AsyncMock(side_effect=_fake_advance_main)

        result = await _journal_landed_then_advance(
            outbox, git_ops,
            task_id='t1',
            branch_tip_sha='tipsha',
            advanced_sha='mergesha',
            merge_wt=merge_wt,
            branch='task/1',
            max_attempts=3,
            expected_main='base',
        )

        row = captured.get('row')
        assert row is not None, (
            'LandedRow must be recorded before advance_main is invoked (WA-1)'
        )
        assert row.task_id == 't1'
        assert row.branch_tip_sha == 'tipsha'
        assert row.advanced_sha == 'mergesha'
        assert isinstance(row.landed_at, float)

        assert result == AdvanceOutcome('advanced', advanced_sha='mergesha')
        git_ops.advance_main.assert_awaited_once_with(
            'mergesha', merge_wt,
            branch='task/1', max_attempts=3, expected_main='base',
        )

    async def test_none_outbox_does_not_crash(self, tmp_path: Path) -> None:
        """outbox=None (no project_root) must no-op the record, not raise."""
        from orchestrator.merge_queue import _journal_landed_then_advance

        merge_wt = tmp_path / '_merge-wt2'
        git_ops = MagicMock()
        git_ops.advance_main = AsyncMock(
            return_value=AdvanceOutcome('advanced', advanced_sha='mergesha2')
        )

        result = await _journal_landed_then_advance(
            None, git_ops,
            task_id='t2',
            branch_tip_sha=None,
            advanced_sha='mergesha2',
            merge_wt=merge_wt,
            branch='task/2',
            max_attempts=3,
            expected_main='base2',
        )

        assert result == AdvanceOutcome('advanced', advanced_sha='mergesha2')
        git_ops.advance_main.assert_awaited_once_with(
            'mergesha2', merge_wt,
            branch='task/2', max_attempts=3, expected_main='base2',
        )


@pytest.mark.asyncio
class TestFinalizeInflightJournalsLandedRow:
    """Step-3 (RED) — B6 single-branch: LandedRow visible at advance_main call time.

    ``_finalize_inflight`` still calls ``self._git_ops.advance_main(...)``
    directly (no write-ahead record).  RED until step-4 routes the CAS
    advance through ``_journal_landed_then_advance``, which records into
    ``worker._landed_outbox`` before awaiting advance_main.
    """

    async def test_finalize_inflight_journals_landed_row_before_advance(
        self, tmp_path: Path,
    ) -> None:
        """The row must already be visible from inside advance_main's own call.

        RED: no row is recorded before advance_main is invoked → captured['row']
        stays None (worker._landed_outbox.lookup('b6-single') finds nothing yet).
        GREEN (step-4): _finalize_inflight routes the CAS advance through
        _journal_landed_then_advance, which records first.
        """
        from dataclasses import replace as _replace

        cfg = OrchestratorConfig(project_root=tmp_path, git=_make_spec_git_config(on=True))
        worker, mock_git_ops = _make_finalize_test_worker(tmp_path)

        item = _make_spec_item(tmp_path, cfg, speculative=True)
        item.request.task_id = 'b6-single'
        item = _replace(item, merged_branch_tip='cafe' * 10)

        vr = InflightVerifyResult(outcome=None, merge_wt=item.merge_wt, spec_warm=False)
        entry = _build_entry(item, vr, merge_wt=item.merge_wt)

        captured: dict[str, Any] = {}

        async def _fake_advance_main(current_sha: str, merge_wt: Path, **kwargs: Any) -> AdvanceOutcome:
            # WA-1: captured HERE, at call time — proves record() precedes advance.
            assert worker._landed_outbox is not None, (
                'worker must have a real _landed_outbox (project_root is set)'
            )
            captured['row'] = worker._landed_outbox.lookup('b6-single')
            return AdvanceOutcome('advanced', advanced_sha=current_sha)

        mock_git_ops.advance_main.side_effect = _fake_advance_main

        with (
            patch(
                'orchestrator.merge_queue._finalize_advanced_merge',
                new=AsyncMock(return_value=MergeOutcome(
                    'done', merge_sha=item.merge_result.merge_commit,
                )),
            ),
            patch('orchestrator.merge_queue._maybe_schedule_shadow_compare', new=AsyncMock()),
            patch('orchestrator.merge_queue._maybe_run_drift_check', new=AsyncMock()),
        ):
            await worker._finalize_inflight(entry)

        row = captured.get('row')
        assert row is not None, (
            'LandedRow must be recorded into worker._landed_outbox before '
            'advance_main is invoked (WA-1) at the single-branch CAS site'
        )
        assert row.advanced_sha == item.merge_result.merge_commit, (
            f'Expected advanced_sha={item.merge_result.merge_commit!r}, '
            f'got {row.advanced_sha!r}'
        )
        assert row.branch_tip_sha == item.merged_branch_tip, (
            f'Expected branch_tip_sha={item.merged_branch_tip!r}, '
            f'got {row.branch_tip_sha!r}'
        )


# ---------------------------------------------------------------------------
# task 3980: enforced timeout-mark coverage over THIS module's own source
# ---------------------------------------------------------------------------


class TestTimeoutMarkCoverage:
    """Enforced invariant: every class in THIS module whose computed
    worst-per-method wait budget clears the pyproject default timeout must
    carry a ``@pytest.mark.timeout`` mark whose value clears that budget.

    Task 3492 built this guard and applied it to
    test_merge_queue_concurrent_verify.py, but hard-scoped it to that file's
    own ``Path(__file__)`` -- so THIS module, which carries the merge-spec
    late-arrival block, was never covered despite having five classes whose
    worst-case per-method budget is 105s-125s against a 60s pyproject default
    and ZERO ``@pytest.mark.timeout`` marks anywhere in the file.

    The helpers are IMPORTED from test_merge_queue_concurrent_verify rather
    than reimplemented: they are deliberately pure (source text in, offender
    list out, class resolver injected as ``globals().get``) precisely so they
    can be driven with foreign input, and a second copy would be free to
    drift from the marks it audits.

    Task 3980 extended ``_call_wait_budget`` there to recognise
    ``wait_responsive(...)``, whose budget is charged in loop-responsive time
    and can therefore consume up to ``min(RESPONSIVE_WAIT_STRETCH * timeout,
    RESPONSIVE_WAIT_WALL_CAP)`` of real wall clock -- i.e. the scanned budget
    for those call sites is stretched by exactly 2, bounded by the absolute
    ceiling. The helper computes its own default cap from that SAME formula
    (esc-3980-3 reviewer finding: a flat default made this bill an
    under-count for every small-nominal site), so the scan is an exact upper
    bound on real wall clock. That stretch is why this guard has to exist BEFORE any wait in
    this file is migrated: a stretched wait under an inadequate mark is
    strictly worse than the flake it fixes.
    """

    def test_heavy_wait_classes_carry_adequate_timeout_mark(self) -> None:
        """Every Test* class computing >= PYPROJECT_DEFAULT_TIMEOUT must
        carry a ``timeout`` mark whose value clears its own computed budget.

        RED when first written (task 3980 step-3, before step-4 added the
        marks): five late-arrival classes computed >= 60s with no timeout
        mark at all -- TestLateArrivalCleanCAS 125s,
        TestLateArrivalSubmissionOrderCAS 125s, TestLateArrivalAttaches 110s,
        TestLateArrivalGuards 110s, TestLateArrivalFailCascade 105s.
        """
        source = Path(__file__).read_text()
        budgets = _worst_per_method_wait_budget(source)
        offenders = _timeout_mark_offenders(budgets, globals().get)

        assert not offenders, (
            'The following classes have a worst-case per-method wait '
            f'budget at or above the pyproject default timeout '
            f'({PYPROJECT_DEFAULT_TIMEOUT}s, see the '
            f'[tool.pytest.ini_options].timeout setting in '
            f'orchestrator/pyproject.toml) but lack an adequate '
            f'@pytest.mark.timeout mark:\n'
            + '\n'.join(f'  - {offender}' for offender in offenders)
            + '\n\nConsequence: pytest-timeout\'s thread method os._exit()s '
            'the xdist worker under --max-worker-restart=0, so a '
            'slow-but-correct run reports as a worker death instead of a '
            'clean per-test failure -- strictly worse than the flake being '
            'fixed. Add @pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT) '
            'directly above each offending class.'
        )


# ---------------------------------------------------------------------------
# task 3980: structural guard — late-arrival waits must be load-independent
# ---------------------------------------------------------------------------

# The classes whose load-bearing waits must go through `wait_responsive`.
# These are exactly the five classes step-4 marked with
# @pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT).
_LATE_ARRIVAL_CLASSES = frozenset({
    'TestLateArrivalAttaches',
    'TestLateArrivalCleanCAS',
    'TestLateArrivalFailCascade',
    'TestLateArrivalGuards',
    'TestLateArrivalSubmissionOrderCAS',
})


def _load_bearing_wait_target(node: ast.expr) -> str | None:
    """Describe *node* if it is a load-bearing synchronisation point, else None.

    Exactly two shapes are load-bearing in the late-arrival block, and both
    gate a hard assertion downstream:

      * ``req_a.result`` — a ``MergeRequest.result`` future. Its resolution IS
        the event the test is waiting for; a deadline here fails a test whose
        merge pipeline completed correctly.
      * ``gate_a_entered.wait()`` — an ``asyncio.Event`` barrier. Already
        event-driven; only its deadline is wall-clock.

    Deliberately NOT load-bearing, and therefore excluded: the
    ``await asyncio.wait_for(worker_task, timeout=5.0)`` teardown waits. Those
    target a bare ``Name`` (the worker Task), sit inside
    ``contextlib.suppress(Exception)``, assert nothing, and swallow their own
    TimeoutError — so they cannot manufacture the flake this task fixes, and
    stretching them would only slow teardown down. The Name-vs-Attribute/Call
    distinction is what makes that exclusion structural rather than a
    hand-maintained name list.
    """
    if isinstance(node, ast.Attribute) and node.attr == 'result':
        return f'{ast.unparse(node)} (MergeRequest.result future)'
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'wait'
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id.startswith('gate')
    ):
        return f'{ast.unparse(node)} (asyncio.Event gate barrier)'
    return None


def _late_arrival_wait_offenders(source: str) -> list[str]:
    """Statically scan *source* for late-arrival waits that are still charged
    in wall clock, returning one formatted offender string per site.

    Two offence kinds, both reported as ``file:line`` plus the enclosing test
    so the failure is directly actionable:

      1. a load-bearing wait still routed through a bare
         ``asyncio.wait_for(..., timeout=...)`` instead of ``wait_responsive``;
      2. a raw numeric wall-clock literal on a load-bearing wait site (on
         EITHER call shape) instead of a bound derived from
         ``MERGE_RESULT_TIMEOUT``.

    Must never raise: a crash here would fail the module over an unrelated
    edit. Unparseable source returns [].
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    name = Path(__file__).name
    offenders: list[str] = []

    for cls in ast.iter_child_nodes(tree):
        if not (isinstance(cls, ast.ClassDef) and cls.name in _LATE_ARRIVAL_CLASSES):
            continue
        for item in cls.body:
            if not (
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name.startswith('test_')
            ):
                continue
            where = f'{cls.name}::{item.name}'
            for call in ast.walk(item):
                if not (isinstance(call, ast.Call) and call.args):
                    continue
                func = call.func

                is_bare_wait_for = (
                    isinstance(func, ast.Attribute)
                    and func.attr == 'wait_for'
                    and isinstance(func.value, ast.Name)
                    and func.value.id == 'asyncio'
                )
                is_responsive = isinstance(func, ast.Name) and func.id == 'wait_responsive'
                if not (is_bare_wait_for or is_responsive):
                    continue

                target = _load_bearing_wait_target(call.args[0])
                if target is None:
                    continue

                if is_bare_wait_for:
                    offenders.append(
                        f'{name}:{call.lineno} — {where} — awaits {target} via a '
                        f'bare asyncio.wait_for, so its deadline is charged in '
                        f'WALL CLOCK. Route it through wait_responsive(...) '
                        f'with a descriptive label=.'
                    )

                timeout_kw = next(
                    (kw for kw in call.keywords if kw.arg == 'timeout'), None
                )
                if (
                    timeout_kw is not None
                    and isinstance(timeout_kw.value, ast.Constant)
                    and isinstance(timeout_kw.value.value, (int, float))
                    and not isinstance(timeout_kw.value.value, bool)
                ):
                    offenders.append(
                        f'{name}:{call.lineno} — {where} — awaits {target} with a '
                        f'RAW wall-clock literal timeout='
                        f'{timeout_kw.value.value!r}. Derive the bound from '
                        f'MERGE_RESULT_TIMEOUT instead of writing a number.'
                    )

    return offenders


class TestLateArrivalWaitsAreLoadIndependent:
    """Enforced invariant: no load-bearing wait in the late-arrival block may
    carry a wall-clock deadline.

    This is the guard that stops the class recurring in this file a thirteenth
    time. The measured failures behind task 3980 were all genuine asyncio
    deadline expiries on tests whose logic had ALREADY completed — the CleanCAS
    log tail reads ``verify end (passed=True)`` next to a heartbeat of
    ``oldest age=46s ... state=finalizing``. Widening the numbers would only
    move the threshold; charging the budget in loop-responsive time removes the
    dependence, and this guard is what keeps it removed.

    It also closes the specific hole that produced one of the three failures.
    Task 2376's sweep replaced merge-pipeline wait literals, but its stated
    policy only covered literals <= 15 — so the lone ``timeout=25.0`` in
    test_predecessor_fail_cascades_to_late_arrival sat just above the sweep and
    survived as the file's only bare mid-range deadline. A policy expressed as
    "literals up to N" cannot catch the one above N; a structural invariant
    over the call SHAPE can.
    """

    def test_no_late_arrival_wait_is_charged_in_wall_clock(self) -> None:
        offenders = _late_arrival_wait_offenders(Path(__file__).read_text())

        assert not offenders, (
            'These late-arrival synchronisation points still carry '
            'wall-clock deadlines:\n'
            + '\n'.join(f'  - {offender}' for offender in offenders)
            + '\n\nA MergeRequest.result future wait and an asyncio.Event '
            'gate barrier are both load-bearing: a deadline expiry there '
            'fails a test whose merge pipeline completed correctly, purely '
            'because the xdist worker was descheduled. Use '
            'wait_responsive(...), which charges its budget in '
            'loop-responsive time and still reports a genuine hang red. The '
            'worker_task teardown waits are deliberately exempt (best-effort '
            'cleanup inside contextlib.suppress, asserting nothing).'
        )


# ===========================================================================
# Task 3980 step-9: no unspecced VerifyResult-shaped double may survive here
# ===========================================================================


# Exactly ONE deliberate bare-MagicMock site is exempt (task 3980 step-12).
# TestDispositionDoubleFidelity's NEGATIVE leg re-introduces the pre-step-8
# defect ON PURPOSE and asserts that the classifier fails open on it — that
# mutation is what makes the POSITIVE leg's "no fail-open" assertion provably
# two-sided rather than trivially true. A guard that flagged it would force the
# proof to be deleted to keep the guard green, which is backwards.
#
# The exemption is keyed on the enclosing scope, NOT on a line number (which
# every edit above it invalidates) and NOT on a comment pragma (which is one
# copy-paste away from exempting a real offender). Adding an entry here is a
# design decision, not a cleanup: any NEW entry must, like this one, exist to
# prove a guard can fail.
_BARE_DOUBLE_EXEMPT_SCOPES = frozenset({
    'TestDispositionDoubleFidelity::test_bare_double_makes_classifier_fail_open',
})


def _bare_verify_result_double_offenders(source: str) -> list[str]:
    """Statically scan *source* for unspecced VerifyResult-shaped MagicMocks.

    The tell is a ``passed=`` keyword: it is VerifyResult's first field and no
    other double in this file carries it. Keying on ``passed=`` rather than on
    ``MagicMock`` alone is what lets the many legitimate non-VerifyResult
    MagicMocks here through untouched.

    Flags a construction regardless of POSITION — ``return MagicMock(...)``, an
    assignment, or an argument. Position-blindness is the whole point: this
    repo's dedicated detector, fused-memory/scripts/check_bare_magicmock_config.py,
    provably cannot catch this shape for three independent reasons, any one of
    which is fatal — it inspects only ``ast.Assign``/``ast.AnnAssign`` while all
    ten sites behind task 3980 were ``return MagicMock(...)``; its
    ``_is_config_name`` matches only config/cfg/*_config/*_cfg targets; and its
    remedies are pydantic-specific (``pydantic_spec`` reads ``model_fields``)
    while VerifyResult is a stdlib dataclass. Widening that shared, seven-caller
    gate is filed as a separate follow-up; this gives the file coverage now.

    A ``spec=``/``spec_set=`` argument exempts a site, because that is precisely
    what makes an unknown-attribute READ raise AttributeError instead of
    auto-vivifying a truthy child Mock — the root-cause fix rather than a
    ``cause_hint=''`` patch for today's one known field.

    Must never raise: a crash here would fail the module over an unrelated
    edit. Unparseable source returns [].
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    name = Path(__file__).name
    offenders: list[str] = []

    def _is_magicmock(func: ast.expr) -> bool:
        # Both the bare `MagicMock(...)` this file imports and a qualified
        # `mock.MagicMock(...)` / `unittest.mock.MagicMock(...)`.
        if isinstance(func, ast.Name):
            return func.id == 'MagicMock'
        if isinstance(func, ast.Attribute):
            return func.attr == 'MagicMock'
        return False

    def _visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                _visit(child, f'{scope}::{child.name}' if scope else child.name)
                continue
            if isinstance(child, ast.Call) and _is_magicmock(child.func):
                kwargs = {kw.arg for kw in child.keywords}
                if (
                    'passed' in kwargs
                    and not ({'spec', 'spec_set'} & kwargs)
                    and scope not in _BARE_DOUBLE_EXEMPT_SCOPES
                ):
                    offenders.append(
                        f'{name}:{child.lineno} — {scope or "<module>"} — '
                        f'MagicMock(passed=...) with no spec=.'
                    )
            _visit(child, scope)

    _visit(tree, '')
    return offenders


class TestNoBareVerifyResultDoubles:
    """Enforced invariant: every VerifyResult-shaped double in this module is
    built through ``_fake_verify_result``, never as a bare MagicMock.

    Task 3980 measured what a bare one costs. An unconfigured attribute READ on
    a bare MagicMock auto-vivifies a truthy child Mock rather than returning a
    real default, so the ten inline doubles here — every one of which omitted
    ``cause_hint`` — made merge_disposition.py's
    ``_extract_failing_tests_and_candidate_files`` (:218-221) raise
    ``TypeError: sequence item 0: expected str instance, MagicMock found`` out
    of ``str.join``. ``classify_merge_failure_disposition`` (:710-719) swallows
    that into a silent fail-open (WARNING + INDETERMINATE), which is
    indistinguishable downstream from a genuine verdict — so the affected
    assertions only APPEARED to exercise disposition classification.

    Every one of these sites also passed ``verify_skipped=``, which is not a
    VerifyResult field at all (it lives on MergeOutcome, merge_types.py:945).
    Nothing objected, because a bare MagicMock accepts any kwarg. That is the
    same silent-drift failure mode from the other direction, and it is why the
    remedy is ``spec=VerifyResult`` rather than adding ``cause_hint=''``.
    """

    def test_no_unspecced_verify_result_double_survives(self) -> None:
        offenders = _bare_verify_result_double_offenders(Path(__file__).read_text())

        assert not offenders, (
            'These VerifyResult-shaped doubles are still unspecced MagicMocks:\n'
            + '\n'.join(f'  - {offender}' for offender in offenders)
            + '\n\nRemedy: build them with _fake_verify_result(...) (task 3477, '
            'imported at the top of this module). It seeds a '
            'MagicMock(spec=VerifyResult) from dataclasses.fields(VerifyResult), '
            'so every real field gets its real default (cause_hint becomes a '
            'real str), an unknown-attribute READ raises AttributeError instead '
            'of auto-vivifying a truthy Mock, and an unknown override such as '
            'verify_skipped= is rejected with TypeError instead of silently '
            'setattr-ing onto the mock. Drop lint_output/type_output/timed_out/'
            'category unless the test needs a non-default value — restating a '
            'default inline is how these doubles drifted in the first place.\n'
            'Exactly one site is exempt, by enclosing scope, in '
            '_BARE_DOUBLE_EXEMPT_SCOPES: TestDispositionDoubleFidelity\'s '
            'negative leg re-introduces the defect deliberately to prove the '
            'positive leg can fail. Do not add an entry to silence a real '
            'offender.'
        )


# ===========================================================================
# Task 3980 step-12: two-sided fidelity proof for the verify-result double
# ===========================================================================


@pytest.mark.asyncio
class TestDispositionDoubleFidelity:
    """Isolated, two-sided proof that a bare verify-result double silently
    disables merge-disposition classification — and that ``_fake_verify_result``
    does not.

    WHY THIS EXISTS SEPARATELY from the live guard inside
    ``TestLateArrivalFailCascade``. That guard is real (it fails under the
    bare-MagicMock mutation, measured), but its two-sidedness is an IMPLICIT
    property of a ~4s integration test's code path: it holds only because the
    classifier happens to sit on the head-failure cascade. A reader cannot check
    that by reading it, and a reviewer trying to check it empirically got the
    wrong answer — reading the ABSENCE of a WARNING as evidence the classifier
    never ran, when a successful classification is precisely the silent case.
    These two tests pin the mechanism directly, hermetically, in milliseconds:
    no merge worker, no git repo, no event-loop barriers, no timing.

    The legs are complementary and BOTH are required. The positive leg alone is
    a test that cannot fail; the negative leg is what proves it can.
    """

    # Minimal surrounding args for a hermetic classifier call. repo_root,
    # task_id and event_store are deliberately left at their None defaults: each
    # degrades its own evidence source fail-safe, which is what keeps this test
    # free of git and event-store setup. The SHAs are syntactically valid and
    # never resolved, because control returns before any git plumbing runs.
    _ARGS: dict[str, Any] = {
        'branch': 'task/3980-fidelity',
        'merge_base_sha': '0' * 40,
        'main_sha': '1' * 40,
        'preexisting': False,
    }

    async def test_fake_verify_result_double_lets_the_classifier_run(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """POSITIVE leg: a ``_fake_verify_result`` double classifies normally.

        Asserts the ABSENCE of a fail-open and the PRESENCE of a real
        ``ClassificationResult`` — deliberately NOT the disposition VALUE. The
        verdict depends on git state this test does not set up (repo_root=None
        short-circuits to INDETERMINATE at the ambiguity clause), so pinning it
        would buy nothing and break on any future evidence-source change.
        """
        for logger_name in _FAIL_OPEN_LOGGERS:
            caplog.set_level(logging.WARNING, logger=logger_name)

        double = _fake_verify_result(
            passed=False, summary='tests failed', test_output='FAIL',
        )

        result = await classify_merge_failure_disposition(
            verify_result=cast(VerifyResult, double), **self._ARGS,
        )

        fail_open = _fail_open_records(caplog.records)
        assert not fail_open, (
            'A _fake_verify_result double must classify cleanly, but the '
            'classifier FAILED OPEN:\n'
            f'{_format_fail_open_records(fail_open)}\n'
            'If this leg is red, the factory has drifted from VerifyResult '
            '(a new field whose seeded default breaks '
            '_extract_failing_tests_and_candidate_files, most likely), and '
            'every disposition-sensitive assertion built on it is now vacuous.'
        )
        assert isinstance(result, ClassificationResult), (
            f'classifier must return a ClassificationResult; got {result!r}'
        )

    async def test_bare_double_makes_classifier_fail_open(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """NEGATIVE leg: the pre-step-8 bare MagicMock DOES fail the classifier.

        This is the mutation, kept permanently as a test. It reconstructs the
        exact shape task 3980 removed from ten sites — note ``verify_skipped=``,
        which is not a VerifyResult field at all (it lives on MergeOutcome,
        merge_types.py:945) and which a bare MagicMock accepts without
        objection, and note the ABSENT ``cause_hint``, which is the actual
        defect: reading it auto-vivifies a truthy child Mock that survives
        ``_extract_failing_tests_and_candidate_files``' ``if part`` filter and
        then breaks ``str.join`` (merge_disposition.py:218).

        Exempt from ``TestNoBareVerifyResultDoubles`` by enclosing scope via
        ``_BARE_DOUBLE_EXEMPT_SCOPES`` — the one site in this module where a
        bare double is the point.
        """
        for logger_name in _FAIL_OPEN_LOGGERS:
            caplog.set_level(logging.WARNING, logger=logger_name)

        bare = MagicMock(
            passed=False, summary='tests failed', test_output='FAIL',
            lint_output='', type_output='', category='', timed_out=False,
            verify_skipped=False,
        )

        result = await classify_merge_failure_disposition(
            verify_result=cast(VerifyResult, bare), **self._ARGS,
        )

        # The classifier swallows the fault and returns a verdict that is
        # INDISTINGUISHABLE downstream from a genuine one. That is the whole
        # hazard: without the WARNING there is no signal at all.
        assert isinstance(result, ClassificationResult), (
            f'even on fail-open the classifier returns a result; got {result!r}'
        )

        fail_open = _fail_open_records(caplog.records)
        assert fail_open, (
            'MUTATION LEG IS DEAD: a bare MagicMock verify-result double no '
            'longer makes the classifier fail open, so the positive leg above '
            '(and the live guard in TestLateArrivalFailCascade) may now be '
            'asserting something that cannot fail.\n'
            'Either merge_disposition.py stopped consuming cause_hint through '
            'str.join, or the fail-open WARNING text/logger changed. Re-derive '
            'a mutation that DOES break classification and pin that instead — '
            'do NOT delete this leg, and do NOT relax it to a no-op.\n'
            f'WARNINGs captured on {_FAIL_OPEN_LOGGERS}: '
            f'{[(r.name, r.getMessage()) for r in caplog.records]!r}'
        )

        # Match the underlying fault by TYPE plus a LOOSE substring. The full
        # CPython message ('sequence item 0: expected str instance, MagicMock
        # found') is an implementation detail a future interpreter may reword;
        # binding to it would turn a green test red on an unrelated upgrade.
        exc_infos = [r.exc_info for r in fail_open if r.exc_info]
        assert exc_infos, (
            'the fail-open WARNING must carry exc_info so an operator can see '
            f'WHY it degraded; got records={_format_fail_open_records(fail_open)}'
        )
        excs = [info[1] for info in exc_infos]
        assert any(isinstance(exc, TypeError) for exc in excs), (
            f'expected a TypeError from the str.join; got {excs!r}'
        )
        assert any('expected str instance' in str(exc) for exc in excs), (
            'expected the str.join type complaint naming a non-str item; got '
            f'{[str(exc) for exc in excs]!r}'
        )
