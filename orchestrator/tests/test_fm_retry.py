"""Tests for the shared fm-restart retry-window module (orchestrator.fm_retry).

This is the single source consumed by McpSession, Scheduler.set_task_status,
and DeterministicRunner._writeback_deploy_success so that every
orchestrator->fused-memory retry budget spans fm's own restart window
(systemd TimeoutStopSec=90 + observed ~15s start + margin) instead of each
site guessing its own (too-small) budget.
"""

from __future__ import annotations

import pytest

from orchestrator.fm_retry import (
    _FM_RETRY_CAP_SECS,
    _FM_RETRY_MAX_SLEEPS,
    FM_NULL_SENTINEL_URL,
    FM_RESTART_RETRY_WINDOW_SECS,
    fm_retry_backoffs,
    is_fm_null_sentinel,
)


class TestWindowConstant:
    """FM_RESTART_RETRY_WINDOW_SECS is the single shared window."""

    def test_window_is_120_seconds(self):
        assert FM_RESTART_RETRY_WINDOW_SECS == 120.0

    def test_window_covers_stop_plus_start_floor(self):
        # Basis: 90s systemd TimeoutStopSec ceiling + ~15s observed start.
        assert FM_RESTART_RETRY_WINDOW_SECS >= 90 + 15


class TestFmRetryBackoffsPurity:
    """fm_retry_backoffs is a pure function of its injected rng."""

    def test_same_injected_rng_yields_identical_schedule(self):
        def midpoint_rng(lo: float, hi: float) -> float:
            return (lo + hi) / 2

        first = fm_retry_backoffs(rng=midpoint_rng)
        second = fm_retry_backoffs(rng=midpoint_rng)
        assert first == second


class TestFmRetryBackoffsClampIdentity:
    """The final sleep is clamped so the schedule sums exactly to the window."""

    def test_sum_equals_window_at_max_draw_boundary(self):
        schedule = fm_retry_backoffs(rng=lambda lo, hi: hi)
        assert sum(schedule) == pytest.approx(FM_RESTART_RETRY_WINDOW_SECS)

    def test_sum_equals_window_at_min_draw_boundary(self):
        schedule = fm_retry_backoffs(rng=lambda lo, hi: lo)
        assert sum(schedule) == pytest.approx(FM_RESTART_RETRY_WINDOW_SECS)

    def test_max_draw_boundary_schedule_length_is_10(self):
        # Fastest possible growth (always draws the exponential ceiling) is
        # verified reachable in exactly 10 sleeps at the default tunables.
        schedule = fm_retry_backoffs(rng=lambda lo, hi: hi)
        assert len(schedule) == 10

    def test_min_draw_boundary_schedule_length_is_16(self):
        # Slowest possible growth (always draws the exponential floor) is
        # verified reachable in exactly 16 sleeps at the default tunables.
        schedule = fm_retry_backoffs(rng=lambda lo, hi: lo)
        assert len(schedule) == 16

    def test_schedule_not_required_to_be_monotonic(self):
        # The final clamped element is routinely smaller than its
        # predecessor (e.g. max-draw ends [..., 20.0, 9.0]) because it is
        # clamped to whatever remains of the window, not to the raw
        # exponential draw. Assert the (non-monotonic) last-element shape
        # directly instead of a doomed non-decreasing assertion.
        schedule = fm_retry_backoffs(rng=lambda lo, hi: hi)
        assert schedule[-1] < schedule[-2]


class TestFmRetryBackoffsBounds:
    """Every sleep is positive and never exceeds the cap."""

    @pytest.mark.parametrize(
        'rng', [lambda lo, hi: hi, lambda lo, hi: lo], ids=['max-draw', 'min-draw']
    )
    def test_every_element_positive_and_capped(self, rng):
        schedule = fm_retry_backoffs(rng=rng)
        assert all(0 < s <= _FM_RETRY_CAP_SECS for s in schedule)

    @pytest.mark.parametrize(
        'rng', [lambda lo, hi: hi, lambda lo, hi: lo], ids=['max-draw', 'min-draw']
    )
    def test_length_within_max_sleeps(self, rng):
        schedule = fm_retry_backoffs(rng=rng)
        assert len(schedule) <= _FM_RETRY_MAX_SLEEPS


class TestFmRetryBackoffsDefaultAttempts:
    """The default (real-random) schedule beats every old per-site budget."""

    def test_default_attempts_exceed_old_budgets(self):
        # Max draws give the fewest sleeps (10, see above) regardless of the
        # actual random draw, so attempts = len+1 >= 11 robustly — beating
        # the old McpSession (3), scheduler (3), and writeback (6) budgets.
        schedule = fm_retry_backoffs()
        attempts = len(schedule) + 1
        assert attempts > 6


class TestNullSentinel:
    """FM_NULL_SENTINEL_URL + is_fm_null_sentinel — the D8 null-sentinel the
    McpSession transport must fail fast on (task 2880).

    The sentinel is a non-routable address (immediate ECONNREFUSED) that can
    never host a *restarting* fused-memory, so the fm-restart retry window is
    inapplicable to it. This constant is the single authority for the sentinel
    string; evals/profile.py imports it DOWN so the profile value and the
    transport predicate cannot drift.
    """

    def test_sentinel_constant_matches_documented_d8_value(self):
        # The exact D8 value documented in evals/profile.py's EVAL_PROFILE.
        assert FM_NULL_SENTINEL_URL == 'http://127.0.0.1:1'

    def test_recognizes_bare_sentinel(self):
        assert is_fm_null_sentinel('http://127.0.0.1:1') is True

    def test_recognizes_trailing_slash(self):
        # McpSession.__init__ already rstrip('/')-es base_url; the predicate
        # normalizes the same way so a raw endpoint still matches.
        assert is_fm_null_sentinel('http://127.0.0.1:1/') is True

    def test_recognizes_mcp_suffix(self):
        # mcp_call passes memory_url + '/mcp'; the predicate strips it
        # defensively for any caller that hands over a raw /mcp endpoint.
        assert is_fm_null_sentinel('http://127.0.0.1:1/mcp') is True

    def test_real_endpoint_not_misclassified(self):
        # A live fm endpoint must NOT be classified as the sentinel — it keeps
        # the full fm-restart retry window.
        assert is_fm_null_sentinel('http://localhost:8002') is False

    def test_none_is_not_sentinel(self):
        assert is_fm_null_sentinel(None) is False

    def test_empty_string_is_not_sentinel(self):
        assert is_fm_null_sentinel('') is False


class TestFmRetryBackoffsLengthCapGuard:
    """An unreachable window is bounded by max_sleeps, not by the window."""

    def test_unreachable_window_exits_on_max_sleeps_count(self):
        schedule = fm_retry_backoffs(
            1000.0,
            base_secs=1.0,
            cap_secs=20.0,
            max_sleeps=32,
            rng=lambda lo, hi: hi,
        )
        assert len(schedule) == 32
        assert all(s <= 20.0 for s in schedule)
        assert sum(schedule) < 1000.0
