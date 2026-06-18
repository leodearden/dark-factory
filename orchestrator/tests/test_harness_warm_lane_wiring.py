"""Tests for warm-lane pool wiring in Harness.__init__ (task 1788, step-17).

Verifies that Harness constructs GitOps with warm_lane_pool_size=
max_concurrent_tasks when git.warm_lane_pool=True, and with size=0
(pool disabled) when the knob is off.

Step-17: RED — Harness constructs GitOps without warm_lane_pool_size.
Step-18: GREEN — Harness passes warm_lane_pool_size from max_concurrent_tasks.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.harness import Harness
from orchestrator.warm_lane_pool import WarmLanePool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_harness(config: OrchestratorConfig) -> Harness:
    """Construct a Harness with heavy constructors patched out."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(config)


def _make_config(
    *,
    max_concurrent_tasks: int,
    warm_lane_pool: bool,
    tmp_path: Path,
) -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig with the given warm-lane settings."""
    # Create a minimal git repo directory so GitOps doesn't fail on init
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / '.git').mkdir()  # bare minimum to satisfy path checks
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=max_concurrent_tasks,
        git=GitConfig(warm_lane_pool=warm_lane_pool),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHarnessWarmLaneWiring:
    """Harness sizes the pool from max_concurrent_tasks at startup (PRD D9)."""

    def test_pool_sized_from_max_concurrent_tasks(self, tmp_path: Path):
        """With warm_lane_pool=True and max_concurrent_tasks=7, pool.size == 7."""
        config = _make_config(
            max_concurrent_tasks=7,
            warm_lane_pool=True,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_pool is not None, (
            'warm_lane_pool should be a WarmLanePool, not None'
        )
        assert isinstance(harness.git_ops.warm_lane_pool, WarmLanePool)
        assert harness.git_ops.warm_lane_pool.size == 7

    def test_pool_none_when_knob_off(self, tmp_path: Path):
        """With warm_lane_pool=False, pool is None regardless of max_concurrent_tasks."""
        config = _make_config(
            max_concurrent_tasks=7,
            warm_lane_pool=False,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_pool is None, (
            'warm_lane_pool should be None when knob is off'
        )

    def test_pool_size_matches_max_concurrent_tasks_3(self, tmp_path: Path):
        """Pool size tracks max_concurrent_tasks=3 (default-ish value)."""
        config = _make_config(
            max_concurrent_tasks=3,
            warm_lane_pool=True,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.warm_lane_pool is not None
        assert harness.git_ops.warm_lane_pool.size == 3

    def test_pool_none_when_max_concurrent_tasks_is_zero(self, tmp_path: Path):
        """max_concurrent_tasks=0 → pool size=0 → pool is None (always exhausted)."""
        config = _make_config(
            max_concurrent_tasks=0,
            warm_lane_pool=True,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        # size=0 → GitOps treats it as disabled → None
        assert harness.git_ops.warm_lane_pool is None


# ---------------------------------------------------------------------------
# Helpers for spec-pool tests
# ---------------------------------------------------------------------------


def _make_runner(name: str) -> VerifyRunnerConfig:
    """Build a minimal enabled VerifyRunnerConfig."""
    return VerifyRunnerConfig(
        name=name,
        ssh_host=f'{name}.example.com',
        git_remote=f'remote-{name}',
    )


def _make_spec_config(
    *,
    merge_spec_warm_lane_pool: bool,
    verify_runners: list[VerifyRunnerConfig],
    tmp_path: Path,
) -> OrchestratorConfig:
    """Build OrchestratorConfig with the spec knob + verify_runners for K tests."""
    repo = tmp_path / 'repo'
    repo.mkdir(exist_ok=True)
    (repo / '.git').mkdir(exist_ok=True)
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=4,
        git=GitConfig(merge_spec_warm_lane_pool=merge_spec_warm_lane_pool),
        verify_runners=verify_runners,
    )


# ---------------------------------------------------------------------------
# Step-5 RED / Step-6 GREEN — spec pool sized from shared K source
# ---------------------------------------------------------------------------


class TestHarnessSpecPoolWiring:
    """Harness passes merge_spec_warm_lane_pool_size=K to GitOps (step-5 RED, step-6 GREEN).

    K = 1 + len(config.enabled_verify_runners) — the SAME expression as
    speculation_depth passed to SpeculativeMergeWorker — so the spec pool size
    and the worker cap derive from one source and cannot drift.
    """

    def test_spec_pool_sized_from_k_with_runners(self, tmp_path: Path):
        """K=3 (1+2 runners) → spec pool size==3 when knob on."""
        runners = [_make_runner('r1'), _make_runner('r2')]
        config = _make_spec_config(
            merge_spec_warm_lane_pool=True,
            verify_runners=runners,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.spec_warm_lane_pool is not None, (
            'spec_warm_lane_pool should be a WarmLanePool when knob on and K>0'
        )
        assert isinstance(harness.git_ops.spec_warm_lane_pool, WarmLanePool)
        # K = 1 + len(enabled_verify_runners) = 1 + 2 = 3
        expected_k = 1 + len(config.enabled_verify_runners)
        assert harness.git_ops.spec_warm_lane_pool.size == expected_k, (
            f'spec pool size must equal K={expected_k}'
        )

    def test_spec_pool_k1_no_runners(self, tmp_path: Path):
        """K=1 (no remote runners) → spec pool size==1 when knob on."""
        config = _make_spec_config(
            merge_spec_warm_lane_pool=True,
            verify_runners=[],
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.spec_warm_lane_pool is not None
        # K = 1 + 0 = 1
        assert harness.git_ops.spec_warm_lane_pool.size == 1

    def test_spec_pool_none_when_knob_off(self, tmp_path: Path):
        """spec_warm_lane_pool is None when merge_spec_warm_lane_pool=False."""
        runners = [_make_runner('r1'), _make_runner('r2')]
        config = _make_spec_config(
            merge_spec_warm_lane_pool=False,
            verify_runners=runners,
            tmp_path=tmp_path,
        )
        harness = _build_harness(config)

        assert harness.git_ops.spec_warm_lane_pool is None, (
            'spec_warm_lane_pool must be None when knob off'
        )
