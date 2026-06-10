"""Tests for PRD §10 invariant 6(b): warm-vs-cold SHADOW compare.

Task 1710: same-candidate warm-vs-cold shadow compare, test-level diff,
born-at-L2 alarm on divergence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Step-1: cadence predicate _shadow_compare_due
# ---------------------------------------------------------------------------

from orchestrator.merge_queue import (  # noqa: E402
    ShadowCompareState,
    _load_shadow_compare_state,
    _save_shadow_compare_state,
    _shadow_compare_due,
)


class TestShadowCompareDue:
    """Unit tests for the pure _shadow_compare_due predicate."""

    # (a) nightly timer fires regardless of merge count
    def test_nightly_fires_when_elapsed(self) -> None:
        state = ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=0.0)
        # 25 hours elapsed, nightly_interval = 24 h
        now = 25 * 3600.0
        assert _shadow_compare_due(
            state, now, every_n_merges=100, nightly_interval_secs=86400.0
        ) is True

    # (b) merge-count leg fires when count meets threshold
    def test_count_fires_when_n_reached(self) -> None:
        state = ShadowCompareState(merges_since_last_shadow=40, last_shadow_run_at=0.0)
        # Only 1 second elapsed, timer should NOT fire on its own
        now = 1.0
        assert _shadow_compare_due(
            state, now, every_n_merges=40, nightly_interval_secs=86400.0
        ) is True

    def test_count_fires_when_count_exceeds_n(self) -> None:
        state = ShadowCompareState(merges_since_last_shadow=45, last_shadow_run_at=0.0)
        now = 1.0
        assert _shadow_compare_due(
            state, now, every_n_merges=40, nightly_interval_secs=86400.0
        ) is True

    # NOT due when neither threshold is met
    def test_not_due_when_neither_threshold_met(self) -> None:
        state = ShadowCompareState(merges_since_last_shadow=5, last_shadow_run_at=0.0)
        # 1 hour elapsed (not nightly) + only 5 merges (below 40)
        now = 3600.0
        assert _shadow_compare_due(
            state, now, every_n_merges=40, nightly_interval_secs=86400.0
        ) is False

    # "whichever sooner" = OR semantics — either leg alone suffices
    def test_or_semantics_count_only(self) -> None:
        # count leg alone triggers; timer has NOT elapsed
        state = ShadowCompareState(merges_since_last_shadow=40, last_shadow_run_at=0.0)
        now = 100.0  # much less than 86400
        assert _shadow_compare_due(
            state, now, every_n_merges=40, nightly_interval_secs=86400.0
        ) is True

    def test_or_semantics_nightly_only(self) -> None:
        # nightly leg alone triggers; count has NOT reached N
        state = ShadowCompareState(merges_since_last_shadow=3, last_shadow_run_at=0.0)
        now = 90000.0  # > 86400
        assert _shadow_compare_due(
            state, now, every_n_merges=40, nightly_interval_secs=86400.0
        ) is True

    # Count leg disableable by setting every_n_merges=0
    def test_count_leg_disabled_when_zero(self) -> None:
        state = ShadowCompareState(merges_since_last_shadow=1000, last_shadow_run_at=0.0)
        # count=1000 but every_n_merges=0 → count leg is OFF
        now = 1.0
        assert _shadow_compare_due(
            state, now, every_n_merges=0, nightly_interval_secs=86400.0
        ) is False

    def test_count_leg_zero_with_nightly_still_fires(self) -> None:
        # nightly leg still fires even when count leg disabled
        state = ShadowCompareState(merges_since_last_shadow=1000, last_shadow_run_at=0.0)
        now = 90000.0
        assert _shadow_compare_due(
            state, now, every_n_merges=0, nightly_interval_secs=86400.0
        ) is True

    # Nightly leg disableable (plan says nightly_interval_secs<=0 disables it,
    # but Field gt=0 prevents <=0 in production config; we test the predicate directly)
    def test_nightly_leg_disabled_when_zero(self) -> None:
        state = ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=0.0)
        now = 999999.0  # huge elapsed time
        assert _shadow_compare_due(
            state, now, every_n_merges=40, nightly_interval_secs=0
        ) is False

    def test_nightly_leg_disabled_with_count_still_fires(self) -> None:
        # count leg fires even when nightly leg disabled
        state = ShadowCompareState(merges_since_last_shadow=40, last_shadow_run_at=0.0)
        now = 1.0
        assert _shadow_compare_due(
            state, now, every_n_merges=40, nightly_interval_secs=0
        ) is True

    def test_exactly_at_count_threshold(self) -> None:
        # exactly N merges → fires (>= semantics)
        state = ShadowCompareState(merges_since_last_shadow=10, last_shadow_run_at=0.0)
        now = 1.0
        assert _shadow_compare_due(
            state, now, every_n_merges=10, nightly_interval_secs=86400.0
        ) is True

    def test_one_below_count_threshold(self) -> None:
        # N-1 merges → does NOT fire on count leg
        state = ShadowCompareState(merges_since_last_shadow=9, last_shadow_run_at=0.0)
        now = 1.0
        assert _shadow_compare_due(
            state, now, every_n_merges=10, nightly_interval_secs=86400.0
        ) is False

    def test_exactly_at_nightly_threshold(self) -> None:
        # exactly 86400 s elapsed → fires (>= semantics)
        state = ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=0.0)
        now = 86400.0
        assert _shadow_compare_due(
            state, now, every_n_merges=40, nightly_interval_secs=86400.0
        ) is True

    def test_just_below_nightly_threshold(self) -> None:
        state = ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=0.0)
        now = 86399.0
        assert _shadow_compare_due(
            state, now, every_n_merges=40, nightly_interval_secs=86400.0
        ) is False


# ---------------------------------------------------------------------------
# Step-3: persisted cadence state load/save round-trip
# ---------------------------------------------------------------------------


class TestShadowCompareStatePersistence:
    """Unit tests for _load_shadow_compare_state + _save_shadow_compare_state."""

    def test_load_returns_default_when_file_missing(self, tmp_path: Path) -> None:
        state = _load_shadow_compare_state(tmp_path / "nonexistent.json")
        assert state == ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=0.0)

    def test_load_returns_default_on_corrupt_json(self, tmp_path: Path) -> None:
        path = tmp_path / "shadow.json"
        path.write_text("{ not valid json !!!")
        state = _load_shadow_compare_state(path)
        assert state == ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=0.0)

    def test_load_returns_default_on_empty_json_object(self, tmp_path: Path) -> None:
        # Missing keys → fail-safe default
        path = tmp_path / "shadow.json"
        path.write_text("{}")
        state = _load_shadow_compare_state(path)
        assert state == ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=0.0)

    def test_round_trip_preserves_all_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "shadow.json"
        original = ShadowCompareState(merges_since_last_shadow=17, last_shadow_run_at=1234567.89)
        _save_shadow_compare_state(path, original)
        loaded = _load_shadow_compare_state(path)
        assert loaded.merges_since_last_shadow == 17
        assert loaded.last_shadow_run_at == pytest.approx(1234567.89, rel=1e-9)

    def test_round_trip_counter_zero_timestamp_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "shadow.json"
        original = ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=0.0)
        _save_shadow_compare_state(path, original)
        loaded = _load_shadow_compare_state(path)
        assert loaded == original

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "c" / "shadow.json"
        state = ShadowCompareState(merges_since_last_shadow=5, last_shadow_run_at=99.0)
        _save_shadow_compare_state(path, state)
        assert path.exists()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "shadow.json"
        state = ShadowCompareState(merges_since_last_shadow=3, last_shadow_run_at=42.0)
        _save_shadow_compare_state(path, state)
        data = json.loads(path.read_text())
        assert "merges_since_last_shadow" in data
        assert "last_shadow_run_at" in data
        assert data["merges_since_last_shadow"] == 3

    def test_simulated_restart_cadence_count_survives(self, tmp_path: Path) -> None:
        """Simulate 3 separate process restarts; counter must accumulate then reset."""
        path = tmp_path / "shadow.json"
        # Restart 1: load default (0), increment, not due yet, save
        state = _load_shadow_compare_state(path)
        state = ShadowCompareState(
            merges_since_last_shadow=state.merges_since_last_shadow + 10,
            last_shadow_run_at=state.last_shadow_run_at,
        )
        _save_shadow_compare_state(path, state)
        # Restart 2: load (10), increment, still not due
        state = _load_shadow_compare_state(path)
        assert state.merges_since_last_shadow == 10
        state = ShadowCompareState(
            merges_since_last_shadow=state.merges_since_last_shadow + 10,
            last_shadow_run_at=state.last_shadow_run_at,
        )
        _save_shadow_compare_state(path, state)
        # Restart 3: load (20), trigger fires at 20, reset
        state = _load_shadow_compare_state(path)
        assert state.merges_since_last_shadow == 20
        assert _shadow_compare_due(state, 1.0, every_n_merges=20, nightly_interval_secs=86400.0)
        # Reset after trigger
        state = ShadowCompareState(merges_since_last_shadow=0, last_shadow_run_at=9999.0)
        _save_shadow_compare_state(path, state)
        state = _load_shadow_compare_state(path)
        assert state.merges_since_last_shadow == 0
        assert state.last_shadow_run_at == pytest.approx(9999.0)
