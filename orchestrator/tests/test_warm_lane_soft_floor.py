"""Tests for θ warm-lane soft-floor proactive dispatch throttle (task 2443).

θ adds an EARLIER, PROACTIVE soft-floor admission check ahead of the
existing ε hard-floor disk-guard (task 1860): before allocating a NEW
divergent warm lane, consult reify's `warm-lane-disk-guard.sh check --soft`
(a soft floor ABOVE the hard floor). See PRD
reify/docs/prds/warm-lane-pool-sizing-lifecycle.md task θ, contract §9.5,
boundary test B10.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.config import GitConfig


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
