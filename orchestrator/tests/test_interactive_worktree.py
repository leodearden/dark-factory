"""Tests for GitOps.create_interactive_worktree — task α (2010).

Isolated interactive warm-worktree primitive: mints a fresh worktree in a new
_iact-* band and CoW-seeds its build cache by reusing _seed_warm_lane, WITHOUT
ever touching WarmLanePool (isolation invariant I1 — strictly disjoint from
the _lane-* dispatch pool and the _spec-* merge-speculation pool).

Mirrors test_warm_lane_integration_gate.py conventions: a committed seed stub
that creates <lane>/target/seeded.bin (orchestration-observable seededness —
literal filefrag/CoW extent-sharing is a reify/filesystem guarantee, NOT
asserted on a CI tmpfs), an ig-style git-repo fixture, and WarmLanePool
FREE-count / assignments_snapshot / is_lane assertions for the I1 signal.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator import config as orchestrator_config
from orchestrator.config import GitConfig
from orchestrator.git_ops import _run


# ---------------------------------------------------------------------------
# Step-01: config knobs
# ---------------------------------------------------------------------------


class TestInteractiveWorktreeConfigDefaults:
    """RED — config knobs: GitConfig + packaged defaults.yaml carry the three new knobs."""

    def test_gitconfig_defaults(self) -> None:
        """GitConfig() exposes max_interactive_worktrees/interactive_worktree_ttl/iact_prefix."""
        config = GitConfig()
        assert config.max_interactive_worktrees == 2, (
            f'expected default max_interactive_worktrees == 2, '
            f'got {config.max_interactive_worktrees!r}'
        )
        assert config.interactive_worktree_ttl == 86400.0, (
            f'expected default interactive_worktree_ttl == 86400.0, '
            f'got {config.interactive_worktree_ttl!r}'
        )
        assert config.iact_prefix == '_iact-', (
            f"expected default iact_prefix == '_iact-', got {config.iact_prefix!r}"
        )

    def test_packaged_defaults_carry_knobs(self) -> None:
        """orchestrator.config._load_defaults()['git'] carries the same three knobs."""
        defaults = orchestrator_config._load_defaults()
        git_defaults = defaults['git']
        assert git_defaults['iact_prefix'] == '_iact-', (
            f"expected packaged defaults.yaml git.iact_prefix == '_iact-', "
            f"got {git_defaults.get('iact_prefix')!r}"
        )
        assert git_defaults['max_interactive_worktrees'] == 2, (
            f"expected packaged defaults.yaml git.max_interactive_worktrees == 2, "
            f"got {git_defaults.get('max_interactive_worktrees')!r}"
        )
        assert git_defaults['interactive_worktree_ttl'] == 86400.0, (
            f"expected packaged defaults.yaml git.interactive_worktree_ttl == 86400.0, "
            f"got {git_defaults.get('interactive_worktree_ttl')!r}"
        )
