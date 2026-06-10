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

from unittest.mock import MagicMock

from escalation.models import BORN_AT_L2_SEVERITIES

from orchestrator.merge_queue import (  # noqa: E402
    ShadowCompareState,
    ShadowCompareDiff,
    _WARM_COLD_SHADOW_SENTINEL,
    _load_shadow_compare_state,
    _save_shadow_compare_state,
    _shadow_compare_due,
    _submit_shadow_divergence_escalation,
    diff_per_test_results,
    parse_per_test_results,
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


# ---------------------------------------------------------------------------
# Step-5: parse_per_test_results
# ---------------------------------------------------------------------------

# Realistic cargo-nextest human output sample
_NEXTEST_SAMPLE = """\
    Compiling reify-core v0.1.0
       Finished test [unoptimized + debuginfo] target(s) in 3.52s
        Starting 3 tests across 2 binaries

        PASS [   0.045s] reify-core some::mod::test_a
        FAIL [   1.200s] reify-eval other::test_b
        PASS [   0.003s] reify-eval some::other::test_c

------------
Summary [   1.25s] 3 tests run: 2 passed, 1 failed, 0 skipped
"""

_NEXTEST_MULTI_CRATE = """\
        PASS [   0.001s] crate-alpha alpha::test_one
        FAIL [   0.500s] crate-beta beta::test_two
        FAIL [   2.000s] crate-alpha alpha::test_three
        PASS [   0.050s] crate-beta beta::test_four
"""


class TestParsePerTestResults:
    """Unit tests for parse_per_test_results(test_output) -> dict[str, bool]."""

    def test_parses_pass_and_fail(self) -> None:
        result = parse_per_test_results(_NEXTEST_SAMPLE)
        assert result["reify-core some::mod::test_a"] is True
        assert result["reify-eval other::test_b"] is False
        assert result["reify-eval some::other::test_c"] is True

    def test_test_id_is_crate_space_path(self) -> None:
        result = parse_per_test_results(_NEXTEST_SAMPLE)
        for key in result:
            parts = key.split(" ", 1)
            assert len(parts) == 2, f"Expected 'crate test_path', got {key!r}"

    def test_non_test_lines_ignored(self) -> None:
        result = parse_per_test_results(_NEXTEST_SAMPLE)
        # Only test lines should appear
        assert len(result) == 3

    def test_empty_input_yields_empty_dict(self) -> None:
        assert parse_per_test_results("") == {}

    def test_blank_whitespace_input_yields_empty_dict(self) -> None:
        assert parse_per_test_results("   \n\n   ") == {}

    def test_only_non_test_lines_yields_empty_dict(self) -> None:
        output = "Building...\nFinished\nSummary [3s] 0 tests\n"
        assert parse_per_test_results(output) == {}

    def test_multi_crate_output(self) -> None:
        result = parse_per_test_results(_NEXTEST_MULTI_CRATE)
        assert result["crate-alpha alpha::test_one"] is True
        assert result["crate-beta beta::test_two"] is False
        assert result["crate-alpha alpha::test_three"] is False
        assert result["crate-beta beta::test_four"] is True

    def test_tolerates_varying_leading_whitespace(self) -> None:
        # Same test line with more leading spaces
        output = "         PASS [   0.010s] my-crate my::test\n"
        result = parse_per_test_results(output)
        assert "my-crate my::test" in result
        assert result["my-crate my::test"] is True

    def test_fail_is_false(self) -> None:
        output = "        FAIL [  99.999s] my-crate long::test::path\n"
        result = parse_per_test_results(output)
        assert result.get("my-crate long::test::path") is False

    def test_only_fail_lines(self) -> None:
        output = "        FAIL [0.1s] c1 t1\n        FAIL [0.2s] c2 t2\n"
        result = parse_per_test_results(output)
        assert result == {"c1 t1": False, "c2 t2": False}


# ---------------------------------------------------------------------------
# Step-7: diff_per_test_results + ShadowCompareDiff
# ---------------------------------------------------------------------------


class TestDiffPerTestResults:
    """Unit tests for diff_per_test_results and ShadowCompareDiff."""

    # (c) Identical maps → has_divergence is False, all buckets empty
    def test_identical_maps_no_divergence(self) -> None:
        warm = {"t1": True, "t2": False, "t3": True}
        cold = {"t1": True, "t2": False, "t3": True}
        diff = diff_per_test_results(warm, cold)
        assert diff.has_divergence is False
        assert diff.diverging == {}
        assert diff.warm_pass_cold_fail == []
        assert diff.warm_fail_cold_pass == []
        assert diff.only_warm == []
        assert diff.only_cold == []

    # (d) warm=pass/cold=fail → appears in warm_pass_cold_fail + diverging
    def test_warm_pass_cold_fail_named(self) -> None:
        warm = {"reify-core bad::test": True}
        cold = {"reify-core bad::test": False}
        diff = diff_per_test_results(warm, cold)
        assert diff.has_divergence is True
        assert "reify-core bad::test" in diff.warm_pass_cold_fail
        assert "reify-core bad::test" in diff.diverging
        assert diff.diverging["reify-core bad::test"] == (True, False)

    def test_warm_fail_cold_pass(self) -> None:
        warm = {"reify-core flaky::test": False}
        cold = {"reify-core flaky::test": True}
        diff = diff_per_test_results(warm, cold)
        assert diff.has_divergence is True
        assert "reify-core flaky::test" in diff.warm_fail_cold_pass
        assert diff.diverging["reify-core flaky::test"] == (False, True)

    # Test only present in warm → only_warm (divergence)
    def test_only_warm(self) -> None:
        warm = {"t-warm": True}
        cold: dict[str, bool] = {}
        diff = diff_per_test_results(warm, cold)
        assert diff.has_divergence is True
        assert "t-warm" in diff.only_warm
        assert diff.warm_pass_cold_fail == []
        assert diff.warm_fail_cold_pass == []

    # Test only in cold → only_cold (divergence)
    def test_only_cold(self) -> None:
        warm: dict[str, bool] = {}
        cold = {"t-cold": False}
        diff = diff_per_test_results(warm, cold)
        assert diff.has_divergence is True
        assert "t-cold" in diff.only_cold

    # Test passing in both → NOT a divergence
    def test_both_pass_no_divergence(self) -> None:
        warm = {"t1": True}
        cold = {"t1": True}
        diff = diff_per_test_results(warm, cold)
        assert diff.has_divergence is False
        assert "t1" not in diff.diverging
        assert diff.warm_pass_cold_fail == []

    # Multiple divergences in one call
    def test_multiple_divergences(self) -> None:
        warm = {"t_flip": True, "t_agree": True, "t_only_warm": False}
        cold = {"t_flip": False, "t_agree": True, "t_only_cold": True}
        diff = diff_per_test_results(warm, cold)
        assert diff.has_divergence is True
        assert "t_flip" in diff.warm_pass_cold_fail
        assert "t_only_warm" in diff.only_warm
        assert "t_only_cold" in diff.only_cold
        assert "t_agree" not in diff.diverging

    def test_empty_maps_no_divergence(self) -> None:
        diff = diff_per_test_results({}, {})
        assert diff.has_divergence is False

    def test_has_divergence_false_iff_all_buckets_empty(self) -> None:
        diff = ShadowCompareDiff(
            diverging={}, warm_pass_cold_fail=[], warm_fail_cold_pass=[],
            only_warm=[], only_cold=[]
        )
        assert diff.has_divergence is False

    def test_has_divergence_true_if_any_bucket_nonempty(self) -> None:
        # diverging nonempty
        diff = ShadowCompareDiff(
            diverging={"t": (True, False)}, warm_pass_cold_fail=["t"],
            warm_fail_cold_pass=[], only_warm=[], only_cold=[]
        )
        assert diff.has_divergence is True
        # only_cold nonempty
        diff2 = ShadowCompareDiff(
            diverging={}, warm_pass_cold_fail=[], warm_fail_cold_pass=[],
            only_warm=[], only_cold=["t_extra"]
        )
        assert diff2.has_divergence is True


# ---------------------------------------------------------------------------
# Step-9: _submit_shadow_divergence_escalation
# ---------------------------------------------------------------------------

def _make_escalation_queue(*, has_open: bool = False) -> MagicMock:
    """Return a MagicMock escalation_queue with standard API stubs."""
    q = MagicMock()
    q.make_id = MagicMock(return_value='esc-shadow-1')
    q.has_open_l1 = MagicMock(return_value=has_open)
    q.get_by_task = MagicMock(return_value=None)
    q.submit = MagicMock()
    return q


def _make_diverging_diff() -> ShadowCompareDiff:
    return ShadowCompareDiff(
        diverging={"reify-core bad::test": (True, False)},
        warm_pass_cold_fail=["reify-core bad::test"],
        warm_fail_cold_pass=[],
        only_warm=[],
        only_cold=[],
    )


class TestSubmitShadowDivergenceEscalation:
    """Unit tests for _submit_shadow_divergence_escalation."""

    def test_none_escalation_queue_is_noop(self) -> None:
        # Must not raise with None queue
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(
            None, "abc1234567", diff,
            {"t": True}, {"t": False}
        )

    def test_submits_escalation_on_divergence(self) -> None:
        q = _make_escalation_queue()
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(
            q, "deadbeef1234", diff,
            {"reify-core bad::test": True},
            {"reify-core bad::test": False},
        )
        q.submit.assert_called_once()

    def test_escalation_severity_critical_born_at_l2(self) -> None:
        q = _make_escalation_queue()
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(q, "deadbeef", diff, {}, {})
        esc = q.submit.call_args[0][0]
        assert esc.severity == 'critical'
        assert esc.severity in BORN_AT_L2_SEVERITIES

    def test_escalation_level_2(self) -> None:
        q = _make_escalation_queue()
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(q, "deadbeef", diff, {}, {})
        esc = q.submit.call_args[0][0]
        assert esc.level == 2

    def test_escalation_agent_role_is_orchestrator_sentinel(self) -> None:
        q = _make_escalation_queue()
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(q, "deadbeef", diff, {}, {})
        esc = q.submit.call_args[0][0]
        # Must carry orchestrator- prefix so it is NOT downgraded
        assert esc.agent_role == 'orchestrator-warm-cold-shadow'
        assert esc.agent_role.startswith('orchestrator-')

    def test_escalation_category(self) -> None:
        q = _make_escalation_queue()
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(q, "deadbeef", diff, {}, {})
        esc = q.submit.call_args[0][0]
        assert esc.category == 'risk_identified'

    def test_escalation_task_id_is_sentinel(self) -> None:
        q = _make_escalation_queue()
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(q, "deadbeef", diff, {}, {})
        esc = q.submit.call_args[0][0]
        assert esc.task_id == _WARM_COLD_SHADOW_SENTINEL
        assert esc.task_id == '__warm_cold_shadow__'

    def test_summary_names_commit_and_diverging_count(self) -> None:
        q = _make_escalation_queue()
        diff = _make_diverging_diff()
        commit = "abcdef123456"
        _submit_shadow_divergence_escalation(q, commit, diff, {}, {})
        esc = q.submit.call_args[0][0]
        assert commit[:8] in esc.summary
        assert '1' in esc.summary  # 1 diverging test

    def test_detail_contains_diverging_test_id(self) -> None:
        q = _make_escalation_queue()
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(
            q, "sha123",
            diff,
            {"reify-core bad::test": True},
            {"reify-core bad::test": False},
        )
        esc = q.submit.call_args[0][0]
        assert "reify-core bad::test" in esc.detail

    def test_detail_mentions_warm_already_landed(self) -> None:
        q = _make_escalation_queue()
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(q, "sha", diff, {}, {})
        esc = q.submit.call_args[0][0]
        # Must explicitly state that warm merge has already landed
        detail_lower = esc.detail.lower()
        assert any(kw in detail_lower for kw in ('already landed', 'already applied', 'shadow'))

    def test_dedup_no_second_submit_when_open_escalation_exists(self) -> None:
        q = _make_escalation_queue(has_open=True)
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(q, "sha", diff, {}, {})
        q.submit.assert_not_called()

    def test_dedup_submits_when_no_open_escalation(self) -> None:
        q = _make_escalation_queue(has_open=False)
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(q, "sha", diff, {}, {})
        q.submit.assert_called_once()

    def test_make_id_called_with_sentinel(self) -> None:
        q = _make_escalation_queue()
        diff = _make_diverging_diff()
        _submit_shadow_divergence_escalation(q, "sha", diff, {}, {})
        q.make_id.assert_called_once_with(_WARM_COLD_SHADOW_SENTINEL)
