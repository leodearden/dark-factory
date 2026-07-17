"""Tests for variable-depth speculative verify placement (task 2359, sibling
of task 2340's depth telemetry).

Covers:
  step-1  RED   — SpeculationProbeConfig defaults / validation / reloadability
  step-2  GREEN — SpeculationProbeConfig submodel + RELOADABLE_FIELDS registration
  step-3  RED   — select_probe_depth(): byte-identical at fraction=0 + frequency/cycling
  step-4  GREEN — select_probe_depth() core (frequency + depth cycling)
  step-5  RED   — select_probe_depth(): availability fallback (d > available_built_depth)
  step-6  GREEN — availability guard
  step-7  RED   — select_probe_depth(): flake-rate suppression
  step-8  GREEN — suppression guard
  step-9  RED   — SpeculativeMergeWorker._recent_verify_fail_rate() / _record_verify_outcome()
  step-10 GREEN — rolling fail-rate window implementation
  step-11 RED   — SpeculativeMergeWorker._available_built_depth()
  step-12 GREEN — built-depth introspection implementation
  step-13 RED   — SpeculativeMergeWorker._probe_verify_placement(item)
  step-14 GREEN — worker placement implementation
  step-15 RED   — _dispatch_item integration: byte-identical default + probe label
  step-16 GREEN — _dispatch_item integration + _record_verify_outcome wiring

GOAL (PHASE-1, sibling of 2340 calibration): let the EXISTING second verify
slot occasionally target a DEEPER already-built speculative stack (cumulative
0..d, d in probe_depths e.g. {2,3,5,8}) instead of the adjacent depth-1
stack, producing genuine depth>=2 merge_verify records (labelled via task
2340's `depth` field) so analyze_speculation_depth.py prints a multi-point
P(pass|depth) curve. Default OFF (probe_fraction=0.0) and byte-identical.

SCOPE BOUNDARY: this task owns verify PLACEMENT + depth LABELLING only, not
build depth -- see plan.json design_decisions for the full rationale. Under
default speculation_depth (K=2), _available_built_depth() stays low and
every probe safely no-op-falls-back to the unchanged _verify_frontier_depth()
path.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    SpeculationProbeConfig,
)

# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: SpeculationProbeConfig
# ---------------------------------------------------------------------------


class TestSpeculationProbeConfigDefaults:
    """Defaults: probe disabled (fraction=0.0), the four canonical probe
    depths, and a conservative 30% suppression threshold.

    RED until step-2 GREEN adds SpeculationProbeConfig + the
    OrchestratorConfig.speculation_probe field.
    """

    def test_defaults_attached_on_orchestrator_config(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        assert isinstance(cfg.speculation_probe, SpeculationProbeConfig)
        assert cfg.speculation_probe.probe_fraction == 0.0
        assert cfg.speculation_probe.probe_depths == [2, 3, 5, 8]
        assert cfg.speculation_probe.suppress_flake_rate == 0.30

    def test_bare_submodel_defaults(self):
        cfg = SpeculationProbeConfig()
        assert cfg.probe_fraction == 0.0
        assert cfg.probe_depths == [2, 3, 5, 8]
        assert cfg.suppress_flake_rate == 0.30


class TestSpeculationProbeConfigValidation:
    """probe_fraction/suppress_flake_rate are bounded to [0,1]; probe_depths
    must be a non-empty list of positive integers.
    """

    @pytest.mark.parametrize('bad_value', [-0.01, 1.01, -1.0, 2.0])
    def test_probe_fraction_out_of_range_rejected(self, bad_value):
        with pytest.raises(ValidationError):
            SpeculationProbeConfig(probe_fraction=bad_value)

    @pytest.mark.parametrize('bad_value', [-0.01, 1.01, -1.0, 2.0])
    def test_suppress_flake_rate_out_of_range_rejected(self, bad_value):
        with pytest.raises(ValidationError):
            SpeculationProbeConfig(suppress_flake_rate=bad_value)

    @pytest.mark.parametrize('good_value', [0.0, 0.5, 1.0])
    def test_probe_fraction_boundary_values_accepted(self, good_value):
        cfg = SpeculationProbeConfig(probe_fraction=good_value)
        assert cfg.probe_fraction == good_value

    def test_probe_depths_empty_list_rejected(self):
        with pytest.raises(ValidationError):
            SpeculationProbeConfig(probe_depths=[])

    @pytest.mark.parametrize('bad_depths', [[0], [-1], [2, 0, 5], [2, -3]])
    def test_probe_depths_non_positive_entry_rejected(self, bad_depths):
        with pytest.raises(ValidationError):
            SpeculationProbeConfig(probe_depths=bad_depths)

    def test_probe_depths_valid_list_accepted(self):
        cfg = SpeculationProbeConfig(probe_depths=[1, 4, 9])
        assert cfg.probe_depths == [1, 4, 9]

    def test_rejected_at_load_via_orchestrator_config(self, monkeypatch, tmp_path):
        """Mirrors TestMinInflightFloorValidation.test_rejected_at_load_via_orchestrator_config
        (test_config_psi_admission_reload.py): a nested-dict construction (as
        YAML would deserialize into) must also raise.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError):
            OrchestratorConfig(speculation_probe={'probe_fraction': 1.5})  # type: ignore[arg-type]


class TestSpeculationProbeConfigReloadDisposition:
    """Every speculation_probe leaf is green-tier: hot-reloadable without a
    process restart, mirroring the psi_admission/delivered_checks groups.
    """

    def test_leaves_are_reloadable(self):
        expected = {
            'speculation_probe.probe_fraction',
            'speculation_probe.probe_depths',
            'speculation_probe.suppress_flake_rate',
        }
        assert expected <= RELOADABLE_FIELDS

    @pytest.mark.parametrize('leaf', list(SpeculationProbeConfig.model_fields))
    def test_every_leaf_is_reloadable(self, leaf):
        assert f'speculation_probe.{leaf}' in RELOADABLE_FIELDS, (
            f'speculation_probe.{leaf!r} is expected to be green-tier reloadable '
            f'but is missing from RELOADABLE_FIELDS'
        )


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: select_probe_depth() core policy
# ---------------------------------------------------------------------------


class TestSelectProbeDepthByteIdenticalAtZero:
    """probe_fraction<=0.0 -> None unconditionally (byte-identical guarantee).

    Availability/suppression args are deliberately varied/inert-favorable
    (huge available_built_depth, None fail rate) to prove the fraction<=0
    short-circuit alone is what drives the None result.

    RED until step-4 GREEN adds select_probe_depth().
    """

    @pytest.mark.parametrize('round_index', list(range(21)))
    def test_zero_fraction_always_none(self, round_index):
        from orchestrator.merge_queue import select_probe_depth

        result = select_probe_depth(
            probe_fraction=0.0,
            probe_depths=[2, 3, 5, 8],
            round_index=round_index,
            available_built_depth=99,
            recent_fail_rate=None,
            suppress_flake_rate=0.30,
        )
        assert result is None


class TestSelectProbeDepthFrequencyAndCycling:
    """probe_fraction=0.1 fires <= ceil(fraction * N) times over N rounds and
    cycles through probe_depths in order (full coverage of all depths).

    Availability/suppression guards are kept inert here (huge
    available_built_depth, None recent_fail_rate) so only the
    frequency/cycling behaviour is exercised.

    RED until step-4 GREEN adds select_probe_depth().
    """

    def test_frequency_bounded_and_depths_cycle_in_order(self):
        from orchestrator.merge_queue import select_probe_depth

        probe_depths = [2, 3, 5, 8]
        n_rounds = 100
        results = [
            select_probe_depth(
                probe_fraction=0.1,
                probe_depths=probe_depths,
                round_index=i,
                available_built_depth=99,
                recent_fail_rate=None,
                suppress_flake_rate=0.30,
            )
            for i in range(n_rounds)
        ]
        fired = [d for d in results if d is not None]

        # Frequency bound: probe_fraction is an upper bound on firing rate.
        assert len(fired) <= math.ceil(n_rounds * 0.1)
        assert len(fired) > 0, 'expected at least one probe round to fire'

        # Every non-None result is a genuine probe_depths member.
        assert all(d in probe_depths for d in fired)

        # Cycling coverage: consecutive probes advance through probe_depths
        # in order, so with >= len(probe_depths) firings every depth appears.
        assert set(fired) == set(probe_depths)

        # Order check: the first 4 firings visit probe_depths in sequence
        # (period=10 at fraction=0.1, so firings land at round_index
        # 0,10,20,... -- probe_index advances by 1 each firing).
        assert fired[:4] == probe_depths


# ---------------------------------------------------------------------------
# step-5 RED / step-6 GREEN: availability fallback (d > available_built_depth)
# ---------------------------------------------------------------------------


class TestSelectProbeDepthAvailabilityFallback:
    """A probe never fires against a stack shallower than the sampled depth
    -- select_probe_depth() falls back to None (the caller's normal
    _verify_frontier_depth() path) rather than building/rebasing anything to
    satisfy the probe.

    probe_fraction=1.0 makes every round a probe round (period=1), isolating
    the availability guard from the frequency/cycling behaviour covered by
    TestSelectProbeDepthFrequencyAndCycling above.

    RED until step-6 GREEN adds the ``d > available_built_depth`` guard.
    """

    def test_insufficient_built_depth_falls_back_to_none(self):
        from orchestrator.merge_queue import select_probe_depth

        result = select_probe_depth(
            probe_fraction=1.0,
            probe_depths=[8],
            round_index=0,
            available_built_depth=3,
            recent_fail_rate=None,
            suppress_flake_rate=0.30,
        )
        assert result is None

    def test_exactly_sufficient_built_depth_returns_sampled_depth(self):
        from orchestrator.merge_queue import select_probe_depth

        result = select_probe_depth(
            probe_fraction=1.0,
            probe_depths=[8],
            round_index=0,
            available_built_depth=8,
            recent_fail_rate=None,
            suppress_flake_rate=0.30,
        )
        assert result == 8

    def test_more_than_sufficient_built_depth_returns_sampled_depth(self):
        from orchestrator.merge_queue import select_probe_depth

        result = select_probe_depth(
            probe_fraction=1.0,
            probe_depths=[8],
            round_index=0,
            available_built_depth=9,
            recent_fail_rate=None,
            suppress_flake_rate=0.30,
        )
        assert result == 8

    @pytest.mark.parametrize('available_built_depth', [0, 1, 2, 7])
    def test_various_insufficient_depths_fall_back_to_none(self, available_built_depth):
        from orchestrator.merge_queue import select_probe_depth

        result = select_probe_depth(
            probe_fraction=1.0,
            probe_depths=[8],
            round_index=0,
            available_built_depth=available_built_depth,
            recent_fail_rate=None,
            suppress_flake_rate=0.30,
        )
        assert result is None


# ---------------------------------------------------------------------------
# step-7 RED / step-8 GREEN: flake-rate suppression (recent_fail_rate >=
# suppress_flake_rate)
# ---------------------------------------------------------------------------


class TestSelectProbeDepthFlakeSuppression:
    """A high recent per-verify FAIL rate suppresses probing entirely --
    when the speculative stack is already thrashing, spending the second
    slot on a deep probe (rather than the normal adjacent depth-1 verify)
    is not worth the coordination overhead.

    probe_fraction=1.0 (period=1) makes round_index=0 a frequency-eligible
    probe round, isolating suppression from the frequency gate.
    available_built_depth is set comfortably above probe_depths so the
    availability guard (step-6) stays inert.

    RED until step-8 GREEN adds the ``recent_fail_rate >=
    suppress_flake_rate`` guard.
    """

    def test_high_fail_rate_suppresses_probe(self):
        from orchestrator.merge_queue import select_probe_depth

        result = select_probe_depth(
            probe_fraction=1.0,
            probe_depths=[2],
            round_index=0,
            available_built_depth=9,
            recent_fail_rate=0.5,
            suppress_flake_rate=0.30,
        )
        assert result is None

    def test_fail_rate_exactly_at_threshold_suppresses_probe(self):
        """>= is inclusive of the threshold itself."""
        from orchestrator.merge_queue import select_probe_depth

        result = select_probe_depth(
            probe_fraction=1.0,
            probe_depths=[2],
            round_index=0,
            available_built_depth=9,
            recent_fail_rate=0.30,
            suppress_flake_rate=0.30,
        )
        assert result is None

    def test_low_fail_rate_allows_probe(self):
        from orchestrator.merge_queue import select_probe_depth

        result = select_probe_depth(
            probe_fraction=1.0,
            probe_depths=[2],
            round_index=0,
            available_built_depth=9,
            recent_fail_rate=0.1,
            suppress_flake_rate=0.30,
        )
        assert result == 2

    def test_none_fail_rate_allows_probe(self):
        """No rolling-window data yet (e.g. a freshly started worker) must
        not suppress -- only an observed high fail rate suppresses.
        """
        from orchestrator.merge_queue import select_probe_depth

        result = select_probe_depth(
            probe_fraction=1.0,
            probe_depths=[2],
            round_index=0,
            available_built_depth=9,
            recent_fail_rate=None,
            suppress_flake_rate=0.30,
        )
        assert result == 2
