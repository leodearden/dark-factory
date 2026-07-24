"""Tests for task 2988 — EXHAUSTED requeue-cap flip + structural-exhaustion
pool-level L2 (PRD ε / W3, plans/warm-lane-exhaustion-hardening-prd.md).

Two coupled mechanisms closing both failure poles of the 2026-07-22 incident:
  (1) Cap-semantics: a WarmLanePoolExhausted requeue no longer counts against
      the per-task requeue cap (threaded table-driven from the disposition
      flip through TerminalReport -> TaskReport -> Scheduler.record_requeue).
  (2) Structural-exhaustion L2: a pool-GLOBAL consecutive-EXHAUSTED counter on
      GitOps fires a harness-installed born-at-L2 escalation after N trips.

Test classes (added across steps 3/7/9/11 — the record_requeue route unit
test lives in test_retry_cap.py, step-5):
  step-03: TestStructuralExhaustionConfigKnob — the GitConfig threshold field
  step-07: test_warm_lane_pool_exhausted_does_not_count — end-to-end retry-cap
  step-09: TestConsecutiveExhaustedCounter — git_ops counter mechanics
  step-11: TestFileStructuralExhaustionL2 / TestStructuralExhaustionWiring —
           the harness filer, dedup, no-op, and callback install
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


# ---------------------------------------------------------------------------
# step-03: GitConfig.warm_lane_structural_exhaustion_l2_threshold knob
# ---------------------------------------------------------------------------


class TestStructuralExhaustionConfigKnob:
    """GitConfig.warm_lane_structural_exhaustion_l2_threshold (task 2988):
    green-tier/hot-reloadable, default 5 (PRD Open Q1), ge=1.  Mirrors
    warm_lane_drift_l2_threshold's Field conventions; defaults.yaml
    deliberately omits it (the Field default is the single source of truth).
    """

    def test_default_is_five(self):
        from orchestrator.config import GitConfig
        assert GitConfig().warm_lane_structural_exhaustion_l2_threshold == 5

    def test_zero_is_rejected_ge_one(self):
        from pydantic import ValidationError

        from orchestrator.config import GitConfig
        with pytest.raises(ValidationError):
            GitConfig(warm_lane_structural_exhaustion_l2_threshold=0)

    def test_negative_is_rejected_ge_one(self):
        from pydantic import ValidationError

        from orchestrator.config import GitConfig
        with pytest.raises(ValidationError):
            GitConfig(warm_lane_structural_exhaustion_l2_threshold=-1)

    def test_accepts_explicit_positive_override(self):
        from orchestrator.config import GitConfig
        cfg = GitConfig(warm_lane_structural_exhaustion_l2_threshold=2)
        assert cfg.warm_lane_structural_exhaustion_l2_threshold == 2


# ---------------------------------------------------------------------------
# step-07: end-to-end — a WarmLanePoolExhausted requeue does NOT burn the
# per-task retry cap.  The counts_against_requeue_cap flag is threaded, table-
# driven, from the disposition -> TerminalReport -> TaskReport -> record_requeue.
# ---------------------------------------------------------------------------


def _build_task_report(
    outcome,
    *,
    counts_against_requeue_cap: bool = True,
    block_reason: str = 'warm_lane_pool_exhausted',
    cost_usd: float = 1.0,
    steward_cost_usd: float = 0.5,
):
    from orchestrator.harness import TaskReport

    return TaskReport(
        task_id='t1',
        title='T1',
        outcome=outcome,
        cost_usd=cost_usd,
        steward_cost_usd=steward_cost_usd,
        block_reason=block_reason,
        block_detail='pool census: all lanes assigned',
        block_phase='plan',
        counts_against_requeue_cap=counts_against_requeue_cap,
    )


@pytest.mark.asyncio
class TestWarmLanePoolExhaustedDoesNotCount:
    """Task 2988 prescribed observable: driving requeue_cap+2 EXHAUSTED
    (counts_against_requeue_cap=False) REQUEUED reports through
    ``harness._apply_retry_cap`` leaves the scheduler's GENUINE requeue counter
    at 0, the transient counter at 0, and NEVER fires
    ``trigger_retry_cap_exhausted`` — so no retry-cap escalation, no
    reblock-guard arming (the 2026-07-22 incident's L2 storm).  A DONE report
    still clears, and a ``counts_against_requeue_cap=True`` REQUEUED still
    increments (the general retry-cap path is unbroken).
    """

    def _config(self, tmp_path):
        from orchestrator.config import OrchestratorConfig
        return OrchestratorConfig(
            max_per_module=1, requeue_cap=3, project_root=tmp_path,
        )

    async def test_warm_lane_pool_exhausted_does_not_count(self, tmp_path):
        from test_retry_cap import _make_harness

        from orchestrator.workflow import WorkflowOutcome

        config = self._config(tmp_path)
        harness = _make_harness(config)
        trigger = AsyncMock()
        harness.scheduler.trigger_retry_cap_exhausted = trigger  # type: ignore[method-assign]

        n = config.requeue_cap + 2
        results = []
        for _ in range(n):
            results.append(
                await harness._apply_retry_cap(
                    't1',
                    _build_task_report(
                        WorkflowOutcome.REQUEUED,
                        counts_against_requeue_cap=False,
                    ),
                    True,
                )
            )

        # No retry-cap escalation, no reblock arming: trigger never fired.
        assert trigger.await_count == 0
        # Neither the genuine nor the transient counter moved.
        assert harness.scheduler._requeue_counts.get('t1', 0) == 0
        assert harness.scheduler.transient_requeue_count('t1') == 0
        # The caller's requeued flag is preserved (cooldown path unaffected).
        assert all(r is True for r in results)
        # ...but the attempt timeline is preserved for a later genuine failure.
        assert len(harness.scheduler.requeue_history('t1')) == n

    async def test_done_clears_and_counting_requeue_still_increments(self, tmp_path):
        from test_retry_cap import _make_harness

        from orchestrator.workflow import WorkflowOutcome

        config = self._config(tmp_path)
        harness = _make_harness(config)
        harness.scheduler.trigger_retry_cap_exhausted = AsyncMock()  # type: ignore[method-assign]

        # A counts_against_requeue_cap=True REQUEUED DOES increment the genuine
        # counter (guard the general path).
        await harness._apply_retry_cap(
            't1',
            _build_task_report(
                WorkflowOutcome.REQUEUED,
                counts_against_requeue_cap=True,
                block_reason='architect returned empty output',
            ),
            True,
        )
        assert harness.scheduler._requeue_counts['t1'] == 1

        # A non-counting EXHAUSTED requeue interleaves WITHOUT shifting it.
        await harness._apply_retry_cap(
            't1',
            _build_task_report(
                WorkflowOutcome.REQUEUED, counts_against_requeue_cap=False,
            ),
            True,
        )
        assert harness.scheduler._requeue_counts['t1'] == 1

        # DONE clears everything.
        await harness._apply_retry_cap(
            't1', _build_task_report(WorkflowOutcome.DONE), False,
        )
        assert 't1' not in harness.scheduler._requeue_counts
        assert harness.scheduler.requeue_history('t1') == []


# ---------------------------------------------------------------------------
# step-09: git_ops-level consecutive-EXHAUSTED counter + threshold callback.
# A pool-GLOBAL counter on GitOps is incremented at the single EXHAUSTED return
# in ``_acquire_warm_lane_impl``; once it reaches the configured threshold and a
# ``_on_structural_exhaustion`` callback is installed, the callback fires with
# ``(count, census)``.  A successful FRESH acquire (acquire_for reused=False)
# resets the counter to 0.  Mirrors the pool drift-counter mechanics
# (warm_lane_pool.py) — fire-each-at-or-above-threshold; dedup is the harness
# filer's job (step-11/12), NOT the counter's.
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    """Minimal git repo with one commit (mirrors test_warm_lane_pool.py)."""
    from orchestrator.git_ops import _run

    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def wl_git_repo(tmp_path: Path) -> Path:
    """A git repo whose default warm-lane base + pool-storage sentinel exist so
    ``acquire_warm_lane``'s pre-acquire base-health gate sees
    ``WarmBaseHealth.OK`` and the mount-presence guard does not false-trip
    (mirrors test_warm_lane_pool.py::wl_git_repo)."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


def _exhausting_git_ops(wl_git_repo: Path, threshold: int):
    """A GitOps whose size-1 pool is FORCED into exhaustion: ``acquire_for``
    always returns None and ``_try_reclaim_lane_for`` finds no candidate, so
    every ``acquire_warm_lane`` reaches the single EXHAUSTED return.
    """
    from orchestrator.config import GitConfig
    from orchestrator.git_ops import GitOps

    config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        warm_lane_pool=True,
        warm_lane_structural_exhaustion_l2_threshold=threshold,
    )
    git_ops = GitOps(config, wl_git_repo, warm_lane_pool_size=1)
    assert git_ops.warm_lane_pool is not None
    # Forced exhaustion: no FREE lane, no reclaim candidate.
    git_ops.warm_lane_pool.acquire_for = AsyncMock(return_value=None)  # type: ignore[method-assign]
    git_ops._try_reclaim_lane_for = AsyncMock(return_value=None)  # type: ignore[method-assign]
    return git_ops


class TestConsecutiveExhaustedCounter:
    """GitOps ``_consecutive_exhausted`` counter + ``_on_structural_exhaustion``
    callback mechanics (task 2988, PRD ε pole-2).
    """

    def test_increments_each_exhausted_below_threshold_no_fire(
        self, wl_git_repo: Path
    ):
        """Each EXHAUSTED return increments the pool-global counter; below the
        threshold the callback does NOT fire."""
        from orchestrator.git_ops import WarmLaneUnavailable

        threshold = 3
        git_ops = _exhausting_git_ops(wl_git_repo, threshold)
        fires: list = []
        git_ops._on_structural_exhaustion = (
            lambda count, census: fires.append((count, census))
        )

        for i in range(1, threshold):  # 1 .. threshold-1
            result = asyncio.run(git_ops.acquire_warm_lane(f'A{i}', 'HEAD'))
            assert result is WarmLaneUnavailable.EXHAUSTED
            assert git_ops._consecutive_exhausted == i, (
                f'counter must increment on each EXHAUSTED (i={i})'
            )
            assert fires == [], 'callback must NOT fire below the threshold'

    def test_fires_once_at_threshold_with_census(self, wl_git_repo: Path):
        """The threshold-th EXHAUSTED fires the callback exactly once with
        ``(count == threshold, census)`` where census is a WarmLanePoolCensus."""
        from orchestrator.git_ops import WarmLaneUnavailable
        from orchestrator.warm_lane_pool import WarmLanePoolCensus

        threshold = 3
        git_ops = _exhausting_git_ops(wl_git_repo, threshold)
        fires: list = []
        git_ops._on_structural_exhaustion = (
            lambda count, census: fires.append((count, census))
        )

        for i in range(1, threshold + 1):  # exactly `threshold` acquires
            result = asyncio.run(git_ops.acquire_warm_lane(f'A{i}', 'HEAD'))
            assert result is WarmLaneUnavailable.EXHAUSTED

        assert git_ops._consecutive_exhausted == threshold
        assert len(fires) == 1, (
            f'callback must fire exactly once at the threshold, got {len(fires)}'
        )
        count, census = fires[0]
        assert count == threshold
        assert isinstance(census, WarmLanePoolCensus), (
            f'callback must carry the α census, got {type(census)!r}'
        )

    def test_fresh_acquire_resets_counter(self, wl_git_repo: Path):
        """A successful FRESH acquire (acquire_for returns ``(lane, reused=False)``)
        resets the counter to 0.  The downstream seed of the fresh lane faults
        (no seed-warm-lane.sh in the fixture) so acquire ultimately returns
        FAULT, but the reset assignment precedes the downstream work — the
        counter is 0 regardless of the eventual return."""
        threshold = 3
        git_ops = _exhausting_git_ops(wl_git_repo, threshold)
        git_ops._on_structural_exhaustion = lambda count, census: None

        # Build the counter up via forced exhaustion (stays below threshold).
        for i in range(2):
            asyncio.run(git_ops.acquire_warm_lane(f'A{i}', 'HEAD'))
        assert git_ops._consecutive_exhausted == 2

        # A genuinely fresh allocation resets the counter.
        lane = wl_git_repo / '.worktrees' / '_lane-reset'
        git_ops.warm_lane_pool.acquire_for = AsyncMock(  # type: ignore[method-assign]
            return_value=(lane, False)
        )
        asyncio.run(git_ops.acquire_warm_lane('B', 'HEAD'))
        assert git_ops._consecutive_exhausted == 0, (
            'a successful fresh acquire must reset the consecutive-EXHAUSTED counter'
        )
