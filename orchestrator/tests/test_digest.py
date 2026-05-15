"""Tests for orchestrator.digest — pure digest helpers and EWA math.

Task 1327 — AFK hardening: Per-N-escalation digest + EWA escalation/done trip.
"""

from __future__ import annotations

import pytest

import orchestrator.digest as digest


class TestUpdateEwa:
    """Unit tests for digest.update_ewa(prev_ewa, escalations_in_step, done_in_step, alpha)."""

    def test_standard_case(self) -> None:
        """Standard EWA update: prev=0.5, esc=10, done=5, alpha=0.3 → 0.3*(10/5)+0.7*0.5=0.95."""
        result = digest.update_ewa(prev_ewa=0.5, escalations_in_step=10, done_in_step=5, alpha=0.3)
        assert result == pytest.approx(0.95), f"Expected 0.95; got {result}"

    def test_done_zero_uses_denominator_one(self) -> None:
        """done==0 guard: denominator treated as 1; esc=4, done=0, alpha=0.3, prev=0.0 → 1.2.

        Proves zero-done pushes EWA up (not crash, not suppression).
        0.3*(4/1) + 0.7*0.0 = 1.2
        """
        result = digest.update_ewa(prev_ewa=0.0, escalations_in_step=4, done_in_step=0, alpha=0.3)
        assert result == pytest.approx(1.2), f"Expected 1.2; got {result}"

    def test_esc_zero_pulls_toward_zero(self) -> None:
        """esc==0 with done>0 pulls EWA toward zero: esc=0, done=5, prev=1.0, alpha=0.3 → 0.7."""
        result = digest.update_ewa(prev_ewa=1.0, escalations_in_step=0, done_in_step=5, alpha=0.3)
        assert result == pytest.approx(0.7), f"Expected 0.7; got {result}"

    def test_alpha_zero_returns_prev_unchanged(self) -> None:
        """alpha=0.0: EWA is never updated — returns prev unchanged."""
        result = digest.update_ewa(prev_ewa=3.5, escalations_in_step=100, done_in_step=1, alpha=0.0)
        assert result == pytest.approx(3.5), f"Expected 3.5 (prev unchanged); got {result}"

    def test_alpha_one_returns_raw_ratio(self) -> None:
        """alpha=1.0: EWA collapses to the raw ratio (esc/max(done,1))."""
        # esc=8, done=4, alpha=1.0 → 8/4 = 2.0
        result = digest.update_ewa(prev_ewa=99.0, escalations_in_step=8, done_in_step=4, alpha=1.0)
        assert result == pytest.approx(2.0), f"Expected 2.0; got {result}"
