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

KNOWN PHASE-1 LIMITATION (reviewer_comprehensive amendment): a firing probe
relabels depth/main_sha attribution only -- the verify itself still runs
against the DISPATCHED item's own (typically shallower) merge_wt, never
against the probed base's worktree content. See ProbePlacement's docstring
in merge_queue.py for the full caveat this implies for
analyze_speculation_depth.py's per-depth P(pass|depth) consumers, and
TestRunInflightVerifyProbeBaseWiring.test_probe_base_overrides_main_sha's
merge_sha assertion below for the regression lock.
"""

from __future__ import annotations

import asyncio
import collections
import math
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    SpeculationProbeConfig,
)
from orchestrator.merge_queue import (
    DecidedItem,
    InflightEntry,
    InflightVerifyResult,
    MergeRequest,
    RealMergeItem,
    SpeculativeMergeWorker,
)
from orchestrator.merge_types import QueuedBranch
from orchestrator.verify_runner import HostLease

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


class TestSelectProbeDepthFrequencyBoundMidRange:
    """Amendment (reviewer_comprehensive, task 2359): the frequency gate's
    docstring guarantees firing rate <= probe_fraction for EVERY fraction,
    not just the fraction=0.1/1.0 corners the tests above happen to cover.

    ``period = max(1, round(1 / probe_fraction))`` broke that bound across a
    wide mid-to-high range -- e.g. fraction=0.4 rounds ``1/0.4``=2.5 DOWN to
    period 2 (50% observed firing, not <=40%), and fraction in [0.67, 1.0)
    rounds down to period 1 (100% firing). ``math.ceil`` fixes this: period
    is always >= 1/probe_fraction, so the observed rate can never exceed the
    requested fraction. Parametrized over the exact mid-range fractions the
    reviewer called out (0.3, 0.4, 0.5, 0.7).
    """

    @pytest.mark.parametrize('probe_fraction', [0.3, 0.4, 0.5, 0.7])
    def test_observed_firing_rate_never_exceeds_fraction(self, probe_fraction):
        from orchestrator.merge_queue import select_probe_depth

        n_rounds = 1000
        fired = [
            d for d in (
                select_probe_depth(
                    probe_fraction=probe_fraction,
                    probe_depths=[2],
                    round_index=i,
                    available_built_depth=99,
                    recent_fail_rate=None,
                    suppress_flake_rate=0.30,
                )
                for i in range(n_rounds)
            )
            if d is not None
        ]

        observed_rate = len(fired) / n_rounds
        assert observed_rate <= probe_fraction, (
            f'probe_fraction={probe_fraction}: observed firing rate '
            f'{observed_rate} exceeds the requested bound'
        )


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


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN: SpeculativeMergeWorker rolling per-verify FAIL
# rate (_record_verify_outcome / _recent_verify_fail_rate)
# ---------------------------------------------------------------------------


def _make_bare_worker() -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for pure unit tests.

    Copied from test_merge_queue_depth_telemetry.py's ``_make_bare_worker``
    (per-file duplication convention -- see that module's docstring). No
    event loop or real git_ops required.
    """
    return SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())


class TestRecentVerifyFailRate:
    """SpeculativeMergeWorker._record_verify_outcome() /
    _recent_verify_fail_rate() -- the rolling per-verify FAIL-rate window
    feeding select_probe_depth()'s suppression guard (step-7/8 above).

    RED until step-10 GREEN adds the bounded deque + these two methods.
    """

    def test_no_outcomes_recorded_returns_none(self):
        """A freshly-started worker has no flake signal yet -- None, not
        0.0 (select_probe_depth() treats a None rate as "do not suppress").
        """
        worker = _make_bare_worker()
        assert worker._recent_verify_fail_rate() is None

    def test_all_pass_returns_zero(self):
        worker = _make_bare_worker()
        for _ in range(5):
            worker._record_verify_outcome(True)
        assert worker._recent_verify_fail_rate() == 0.0

    def test_all_fail_returns_one(self):
        worker = _make_bare_worker()
        for _ in range(4):
            worker._record_verify_outcome(False)
        assert worker._recent_verify_fail_rate() == 1.0

    def test_mixed_outcomes_returns_fails_over_total(self):
        worker = _make_bare_worker()
        for passed in [True, True, False, True, False]:  # 2 fail / 5 total
            worker._record_verify_outcome(passed)
        assert worker._recent_verify_fail_rate() == pytest.approx(2 / 5)

    def test_window_is_bounded_to_class_constant(self):
        worker = _make_bare_worker()
        assert (
            worker._recent_verify_outcomes.maxlen
            == SpeculativeMergeWorker.RECENT_VERIFY_OUTCOME_WINDOW
        )

    def test_window_bounded_oldest_evicted(self):
        """Recording beyond the window bound evicts the oldest outcomes --
        the rate reflects only the most recent ones (not a lifetime
        average), so a worker's flake signal tracks CURRENT thrash, not
        ancient history.
        """
        worker = _make_bare_worker()
        worker._recent_verify_outcomes = collections.deque(maxlen=3)
        for passed in [False, False, False]:  # would-be rate 1.0
            worker._record_verify_outcome(passed)
        assert worker._recent_verify_fail_rate() == 1.0
        for passed in [True, True, True]:  # evicts all 3 fails
            worker._record_verify_outcome(passed)
        assert worker._recent_verify_fail_rate() == 0.0


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN: SpeculativeMergeWorker._available_built_depth()
# ---------------------------------------------------------------------------


def _make_spec_item(*, speculative: bool) -> RealMergeItem:
    """Build a minimal RealMergeItem for _available_built_depth() tests.

    Mirrors test_merge_queue_resource_audit.py's ``_make_spec_item``
    (per-file duplication convention) -- the built-depth counter under test
    reads only queue membership/isinstance, never any request/merge_result
    field, so a near-bare MagicMock request/merge_result is enough.
    """
    return RealMergeItem(
        request=MagicMock(),
        merge_result=MagicMock(merge_commit='deadbeef01234567890a'),
        merge_wt=Path('_merge-x'),
        base_sha='aabbccdd00000000aaaa',
        speculative=speculative,
    )


def _make_decided_item() -> DecidedItem:
    """Build a minimal DecidedItem (a terminal passthrough -- conflict /
    already_merged / etc -- carries no merge commit and must never count
    toward the built-chain depth).
    """
    return DecidedItem(
        request=MagicMock(),
        immediate_outcome=MagicMock(),
        base_sha='aabbccdd00000000aaaa',
        speculative=False,
    )


def _make_frozen_decided_entry() -> InflightEntry:
    """Wrap a :func:`_make_decided_item` passthrough in an InflightEntry.

    Amendment (reviewer_comprehensive, task 2359): in real operation
    ``_frozen_inflight_entries()`` never actually yields a passthrough (its
    own docstring excludes them), but ``_available_built_depth()`` must not
    assume that -- it should apply the exact same type/commit predicate on
    the frozen scan as it does on the queue scan, so a hypothetical
    passthrough landing here is provably as inert as one in the queue (see
    ``test_decided_passthrough_in_queue_not_counted`` below).
    """
    return InflightEntry(
        item=_make_decided_item(), lease=None, verify_task=MagicMock(),
        merge_wt=None, was_speculative=False,
    )


def _make_frozen_commitless_entry() -> InflightEntry:
    """A frozen entry wrapping a RealMergeItem with a falsy merge_commit.

    Amendment (reviewer_comprehensive, task 2359): locks the invariant that
    ``_available_built_depth()`` and ``_built_depth_tip()`` apply IDENTICAL
    predicates. Before this amendment, ``_available_built_depth()`` counted
    frozen entries unconditionally (no ``isinstance``/commit check at all),
    so a commit-less frozen entry like this one would have inflated the
    count past what ``_built_depth_tip()`` (which DOES require a truthy
    commit) could actually resolve -- silently mis-walking a probe onto the
    wrong (shallower) commit rather than failing closed. See
    ``test_commitless_real_item_in_frozen_not_counted`` below.
    """
    item = RealMergeItem(
        request=MagicMock(),
        merge_result=MagicMock(merge_commit=''),
        merge_wt=Path('_merge-x'),
        base_sha='aabbccdd00000000aaaa',
        speculative=True,
    )
    return InflightEntry(
        item=item, lease=None, verify_task=MagicMock(), merge_wt=None,
        was_speculative=True,
    )


def _make_commitless_spec_item() -> RealMergeItem:
    """A queued RealMergeItem with a falsy merge_commit (mirrors
    :func:`_make_frozen_commitless_entry` for the queue-side scan)."""
    return RealMergeItem(
        request=MagicMock(),
        merge_result=MagicMock(merge_commit=''),
        merge_wt=Path('_merge-x'),
        base_sha='aabbccdd00000000aaaa',
        speculative=True,
    )


class TestAvailableBuiltDepth:
    """SpeculativeMergeWorker._available_built_depth() -- the deepest
    already-built (merged) speculative cumulative stack depth: frozen/
    verifying entries (_frozen_inflight_entries()) plus built-but-
    unverified items still sitting in _verifier_queue, read without
    draining or reordering it (task 2359).

    RED until step-12 GREEN adds the method.
    """

    def test_nothing_built_returns_zero(self):
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: []
        assert worker._available_built_depth() == 0

    def test_frozen_only_returns_frozen_count(self):
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [
            _make_frozen_entry(f'commit-{i}') for i in range(3)
        ]
        assert worker._available_built_depth() == 3

    def test_frozen_plus_queued_speculative_sums(self):
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [
            _make_frozen_entry(f'commit-{i}') for i in range(3)
        ]
        for _ in range(2):
            worker._verifier_queue.put_nowait(_make_spec_item(speculative=True))
        assert worker._available_built_depth() == 5

    def test_decided_passthrough_in_queue_not_counted(self):
        """A DecidedItem (conflict/already_merged passthrough) never
        extends the built chain -- it carries no merge commit.
        """
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: []
        worker._verifier_queue.put_nowait(_make_decided_item())
        assert worker._available_built_depth() == 0

    def test_decided_passthrough_in_frozen_not_counted(self):
        """Amendment (reviewer_comprehensive, task 2359): mirrors the
        queue-side passthrough test above -- a passthrough sitting in the
        FROZEN scan must be equally inert, locking the two scans' predicate
        parity.
        """
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [_make_frozen_decided_entry()]
        assert worker._available_built_depth() == 0

    def test_commitless_real_item_in_frozen_not_counted(self):
        """Amendment (reviewer_comprehensive, task 2359): a RealMergeItem
        with an empty merge_commit must not be counted, matching
        _built_depth_tip()'s truthy-commit requirement -- otherwise
        select_probe_depth() could accept a depth this method claims is
        available but _built_depth_tip() cannot actually resolve.
        """
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [_make_frozen_commitless_entry()]
        assert worker._available_built_depth() == 0

    def test_commitless_real_item_in_queue_not_counted(self):
        """Queue-side counterpart of the frozen-side test above."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: []
        worker._verifier_queue.put_nowait(_make_commitless_spec_item())
        assert worker._available_built_depth() == 0

    def test_none_sentinel_in_queue_not_counted(self):
        """The shutdown sentinel (None) must never be counted."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: []
        worker._verifier_queue.put_nowait(None)
        assert worker._available_built_depth() == 0

    def test_does_not_mutate_or_drain_the_queue(self):
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: []
        worker._verifier_queue.put_nowait(_make_spec_item(speculative=True))
        worker._available_built_depth()
        assert worker._verifier_queue.qsize() == 1

    def test_lockstep_with_built_depth_tip_across_mixed_entries(self):
        """Amendment (reviewer_comprehensive, task 2359): the core
        lockstep invariant -- for ANY mix of real/commit-less/passthrough
        entries, whatever _available_built_depth() reports must be exactly
        what _built_depth_tip() can walk to (never None, never a
        mis-resolved shallower commit). Mixes a commit-less frozen entry
        and a passthrough BEFORE the two genuine built entries to prove the
        walk in _built_depth_tip() and the count in
        _available_built_depth() skip the same non-counting entries in the
        same order.
        """
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [
            _make_frozen_commitless_entry(),
            _make_frozen_decided_entry(),
            _make_frozen_entry('commit-depth-1'),
            _make_frozen_entry('commit-depth-2'),
        ]
        worker._verifier_queue.put_nowait(_make_commitless_spec_item())

        available = worker._available_built_depth()

        assert available == 2
        assert worker._built_depth_tip(available) == 'commit-depth-2'
        assert worker._built_depth_tip(1) == 'commit-depth-1'


# ---------------------------------------------------------------------------
# step-13 RED / step-14 GREEN: SpeculativeMergeWorker._probe_verify_placement()
# ---------------------------------------------------------------------------


def _make_frozen_entry(
    merge_commit: str, *, base_sha: str = 'aabbccdd00000000aaaa',
) -> InflightEntry:
    """Build a frozen InflightEntry wrapping a REAL RealMergeItem with a
    known merge_commit.

    Used both by TestAvailableBuiltDepth above (a real, distinct commit per
    entry -- required since task 2359's amendment made
    ``_available_built_depth()`` apply the same truthy-commit predicate as
    ``_built_depth_tip()``, so an opaque identity-only sentinel would no
    longer count) and by ``_probe_verify_placement()``'s depth-d tip
    resolution below, which needs a real commit string to resolve and
    return.

    *base_sha* defaults to a fixed constant (sufficient when callers only
    care about the entry's merge_commit / count); pass an explicit value to
    form a properly CHAINED sequence of entries for
    check_frozen_prefix_invariant()-based tests.
    """
    item = RealMergeItem(
        request=MagicMock(),
        merge_result=MagicMock(merge_commit=merge_commit),
        merge_wt=Path('_merge-x'),
        base_sha=base_sha,
        speculative=True,
    )
    return InflightEntry(
        item=item, lease=None, verify_task=MagicMock(), merge_wt=None,
        was_speculative=True,
    )


def _make_probe_config(
    *,
    probe_fraction: float,
    probe_depths: list[int] | None = None,
    suppress_flake_rate: float = 0.30,
) -> OrchestratorConfig:
    """Build an OrchestratorConfig carrying a live SpeculationProbeConfig.

    Mirrors test_merge_queue_depth_telemetry.py's ``_make_bare_config()``
    (bare inline construction, no monkeypatched chdir/env needed).
    """
    return OrchestratorConfig(
        speculation_probe=SpeculationProbeConfig(
            probe_fraction=probe_fraction,
            probe_depths=probe_depths if probe_depths is not None else [2, 3, 5, 8],
            suppress_flake_rate=suppress_flake_rate,
        ),
    )


def _make_probe_item(*, speculative: bool, config: OrchestratorConfig) -> RealMergeItem:
    """Build a RealMergeItem carrying a live *config* for
    _probe_verify_placement() (task 2359).

    Only ``.speculative`` and ``.request.config.speculation_probe`` are
    read by the method under test; merge_result/merge_wt/base_sha are
    near-bare placeholders (mirrors ``_make_spec_item`` above).
    """
    req = MergeRequest(
        task_id='t-probe', branch=QueuedBranch.parse('task/t-probe', config.git.branch_prefix), worktree=Path('_wt'),
        pre_rebased=False, task_files=None, module_configs=[], config=config,
        result=MagicMock(),
    )
    return RealMergeItem(
        request=req,
        merge_result=MagicMock(merge_commit='unused'),
        merge_wt=Path('_merge-y'),
        base_sha='aabbccdd00000000aaaa',
        speculative=speculative,
    )


class TestProbeVerifyPlacement:
    """SpeculativeMergeWorker._probe_verify_placement(item) -- the stateful
    wrapper around select_probe_depth() that owns the live round counter,
    reads config off the DISPATCHED item (live, hot-reload-friendly), and
    resolves the probed depth's already-built base commit (task 2359).

    RED until step-14 GREEN adds ProbePlacement + the method.
    """

    def test_non_speculative_item_never_probed(self):
        """The head trust-anchor verify (speculative=False) is never
        probed, regardless of config -- even a probe_fraction=1.0 config
        must not fire for it, and its dispatch must not consume a probe
        round (the counter tracks SECOND-SLOT rounds only).
        """
        worker = _make_bare_worker()
        config = _make_probe_config(probe_fraction=1.0, probe_depths=[2])
        item = _make_probe_item(speculative=False, config=config)

        result = worker._probe_verify_placement(item)

        assert result is None
        assert worker._probe_round_counter == 0

    def test_default_config_byte_identical(self):
        """probe_fraction=0.0 (the default) -> None, unconditionally --
        _dispatch_item's caller falls through to the unchanged
        _verify_frontier_depth() path.
        """
        worker = _make_bare_worker()
        config = _make_probe_config(probe_fraction=0.0)
        item = _make_probe_item(speculative=True, config=config)

        result = worker._probe_verify_placement(item)

        assert result is None

    def test_disabled_config_skips_expensive_introspection(self):
        """probe_fraction<=0.0 short-circuits BEFORE computing
        _available_built_depth()/_recent_verify_fail_rate() -- the
        byte-identical default path must cost nothing beyond the
        item.speculative/cfg.probe_fraction reads (reviewer_comprehensive
        amendment: these two O(n) introspection scans were previously
        computed on every speculative dispatch even though
        select_probe_depth() discarded them via its own probe_fraction
        guard -- byte-identical OUTPUT but not cost).
        """
        worker = _make_bare_worker()
        worker._available_built_depth = MagicMock(return_value=0)  # type: ignore[method-assign]
        worker._recent_verify_fail_rate = MagicMock(return_value=None)  # type: ignore[method-assign]
        config = _make_probe_config(probe_fraction=0.0)
        item = _make_probe_item(speculative=True, config=config)

        result = worker._probe_verify_placement(item)

        assert result is None
        worker._available_built_depth.assert_not_called()
        worker._recent_verify_fail_rate.assert_not_called()

    def test_probe_fires_returns_depth_and_built_tip_base(self):
        """probe_fraction=1.0, probe_depths=[2], a built stack of depth 2,
        and a low recent fail rate -> a placement at depth 2 whose base is
        the depth-2 built cumulative tip (the newest of the two frozen
        entries) -- resolved via the frozen-prefix chain machinery.
        """
        worker = _make_bare_worker()
        entry_1 = _make_frozen_entry('commit-depth-1')
        entry_2 = _make_frozen_entry('commit-depth-2')
        worker._frozen_inflight_entries = lambda: [entry_1, entry_2]
        worker._record_verify_outcome(True)  # recent_fail_rate == 0.0 (low)
        config = _make_probe_config(probe_fraction=1.0, probe_depths=[2])
        item = _make_probe_item(speculative=True, config=config)

        result = worker._probe_verify_placement(item)

        assert result is not None
        assert result.depth == 2
        assert result.base == 'commit-depth-2'

    def test_round_counter_increments_deterministically_across_calls(self):
        """Each speculative-item call advances the round counter by
        exactly 1, so consecutive calls sample consecutive round_index
        values (select_probe_depth()'s frequency/cycling behaviour)."""
        worker = _make_bare_worker()
        entry_1 = _make_frozen_entry('commit-depth-1')
        entry_2 = _make_frozen_entry('commit-depth-2')
        worker._frozen_inflight_entries = lambda: [entry_1, entry_2]
        config = _make_probe_config(probe_fraction=1.0, probe_depths=[2])
        item = _make_probe_item(speculative=True, config=config)

        assert worker._probe_round_counter == 0
        worker._probe_verify_placement(item)
        assert worker._probe_round_counter == 1
        worker._probe_verify_placement(item)
        assert worker._probe_round_counter == 2
        worker._probe_verify_placement(item)
        assert worker._probe_round_counter == 3

    def test_falls_back_to_none_when_built_depth_insufficient(self):
        """Sampled depth exceeds the built stack -> None (fall back to the
        normal _verify_frontier_depth() placement) -- a probe never
        triggers building/rebasing to satisfy itself.
        """
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: []  # nothing built
        config = _make_probe_config(probe_fraction=1.0, probe_depths=[2])
        item = _make_probe_item(speculative=True, config=config)

        result = worker._probe_verify_placement(item)

        assert result is None

    def test_reads_config_live_from_item(self):
        """Config is read from item.request.config on EVERY call (never
        cached at worker-construction time) -- an item carrying a
        probe-enabled config fires even when a PRIOR call on this same
        worker saw a probe-disabled config.
        """
        worker = _make_bare_worker()
        entry_1 = _make_frozen_entry('commit-depth-1')
        entry_2 = _make_frozen_entry('commit-depth-2')
        worker._frozen_inflight_entries = lambda: [entry_1, entry_2]

        off_config = _make_probe_config(probe_fraction=0.0)
        off_item = _make_probe_item(speculative=True, config=off_config)
        assert worker._probe_verify_placement(off_item) is None

        on_config = _make_probe_config(probe_fraction=1.0, probe_depths=[2])
        on_item = _make_probe_item(speculative=True, config=on_config)
        result = worker._probe_verify_placement(on_item)
        assert result is not None
        assert result.depth == 2


# ---------------------------------------------------------------------------
# step-15 RED / step-16 GREEN: _dispatch_item integration
# ---------------------------------------------------------------------------


def _fake_local_allocator() -> MagicMock:
    """A MagicMock HostAllocator stub with a free local slot and a patched
    ``acquire`` returning a fake local HostLease directly -- bypasses the
    real LocalRunner factory closure entirely (merge_wt realness is
    irrelevant to these wiring tests).

    Copied from test_merge_queue_verify_base_invariant.py's
    ``_fake_local_allocator`` (per-file duplication convention).
    """
    allocator = MagicMock()
    allocator.free_host_count.return_value = 1
    allocator.acquire = AsyncMock(
        return_value=HostLease(name='local', runner=MagicMock(), is_local=True),
    )
    return allocator


def _make_dispatch_ready_item(
    *,
    config: OrchestratorConfig,
    merge_wt: Path,
    merge_commit: str,
    base_sha: str = 'item-own-natural-base',
) -> RealMergeItem:
    """Build a dispatch-ready RealMergeItem with a REAL result Future.

    Unlike ``_make_probe_item``'s bare ``MagicMock`` result,
    ``_dispatch_item``'s ``_request_abandoned()`` guard calls
    ``req.result.cancelled()`` -- a MagicMock call is always truthy, which
    would incorrectly short-circuit dispatch down the pre-dispatch-abandon
    path -- so this needs a genuine (never-cancelled) asyncio.Future.
    """
    req = MergeRequest(
        task_id='t-dispatch', branch=QueuedBranch.parse('task/t-dispatch', config.git.branch_prefix), worktree=merge_wt,
        pre_rebased=False, task_files=None, module_configs=[], config=config,
        result=asyncio.get_running_loop().create_future(),
    )
    return RealMergeItem(
        request=req,
        merge_result=MagicMock(merge_commit=merge_commit),
        merge_wt=merge_wt,
        base_sha=base_sha,
        speculative=True,
    )


@pytest.mark.asyncio
class TestDispatchItemProbeWiring:
    """_dispatch_item's depth/probe_base computation (task 2359 step-16) --
    proof via a stand-in ``_run_inflight_verify`` capturing its kwargs
    (mirrors test_merge_queue_verify_base_invariant.py's
    TestDispatchRefreshesLastKnownMainSha harness style: a fake local
    HostAllocator + a fully-replaced ``_run_inflight_verify`` -- no real
    verify/runner/event-store machinery needed since this proves ONLY the
    ``_dispatch_item`` -> ``_run_inflight_verify`` call-site wiring; task
    2340's own tests already exhaustively cover depth's forwarding from
    ``_run_inflight_verify`` through to the emitted ``merge_verify`` event).

    RED until step-16 GREEN wires ``_probe_verify_placement()`` into
    ``_dispatch_item`` and adds the ``probe_base`` param to
    ``_run_inflight_verify``.
    """

    async def test_default_config_byte_identical(self, tmp_path: Path) -> None:
        """probe_fraction=0.0 (default) -> depth == _verify_frontier_depth()
        (the unchanged pre-task-2359 value) and probe_base is None."""
        worker = _make_bare_worker()
        worker._host_allocator = _fake_local_allocator()
        worker._frozen_inflight_entries = lambda: [_make_frozen_entry('commit-x')]

        captured: dict = {}

        async def _fake_run_inflight_verify(item, lease, depth=None, probe_base=None):
            captured['depth'] = depth
            captured['probe_base'] = probe_base
            return InflightVerifyResult(outcome=None, merge_wt=item.merge_wt)

        worker._run_inflight_verify = _fake_run_inflight_verify
        config = _make_probe_config(probe_fraction=0.0)
        item = _make_dispatch_ready_item(
            config=config, merge_wt=tmp_path, merge_commit='c-dispatch',
        )

        entry = await worker._dispatch_item(item)

        assert entry is not None, 'dispatch should succeed with a free local slot'
        assert entry.verify_task is not None
        await entry.verify_task
        assert captured['depth'] == 1
        assert captured['probe_base'] is None

    async def test_probe_fires_dispatches_with_probed_depth_and_base(
        self, tmp_path: Path,
    ) -> None:
        """probe_fraction=1.0, probe_depths=[2], a built stack of depth 2
        -> the dispatched verify is launched with depth=2 against the
        depth-2 built cumulative tip. The dispatched item's OWN base_sha
        (a distinct value from the probed tip) and the pre-existing frozen
        chain are both untouched -- the probe never reorders/rebases any
        in-flight verify (task 1890 invariant).
        """
        worker = _make_bare_worker()
        worker._host_allocator = _fake_local_allocator()
        main_sha = 'aabbccdd00000000aaaa'
        entry_1 = _make_frozen_entry('commit-depth-1', base_sha=main_sha)
        entry_2 = _make_frozen_entry('commit-depth-2', base_sha='commit-depth-1')
        worker._frozen_inflight_entries = lambda: [entry_1, entry_2]

        captured: dict = {}

        async def _fake_run_inflight_verify(item, lease, depth=None, probe_base=None):
            captured['depth'] = depth
            captured['probe_base'] = probe_base
            return InflightVerifyResult(outcome=None, merge_wt=item.merge_wt)

        worker._run_inflight_verify = _fake_run_inflight_verify
        config = _make_probe_config(probe_fraction=1.0, probe_depths=[2])
        item = _make_dispatch_ready_item(
            config=config, merge_wt=tmp_path, merge_commit='c-dispatch',
        )

        entry = await worker._dispatch_item(item)

        assert entry is not None, 'dispatch should succeed with a free local slot'
        assert entry.verify_task is not None
        await entry.verify_task
        assert captured['depth'] == 2
        assert captured['probe_base'] == 'commit-depth-2'
        # The probe must never mutate the dispatched item itself.
        assert entry.item.base_sha == 'item-own-natural-base'
        # And the pre-existing frozen chain must stay structurally healthy.
        assert worker.check_frozen_prefix_invariant(main_sha) == []


@pytest.mark.asyncio
class TestRunInflightVerifyProbeBaseWiring:
    """_run_inflight_verify's new ``probe_base`` parameter (task 2359
    step-16) overrides the ``main_sha`` fed to ``_run_post_merge_verify``'s
    merge-skew classification metadata -- WITHOUT touching ``item`` itself
    (item.base_sha, hence the frozen-prefix chain, is never mutated).

    Mirrors test_merge_queue_depth_telemetry.py's
    TestRunInflightVerifyDepthWiring exactly (same MagicMock git_ops +
    HostLease(is_local=False) + patched ``_run_post_merge_verify``
    capturing kwargs), scoped to the NEW ``probe_base`` -> ``main_sha``
    channel rather than the already-covered ``depth`` channel.

    RED until step-16 GREEN adds the ``probe_base`` parameter.
    """

    async def test_probe_base_overrides_main_sha(self, tmp_path: Path) -> None:
        from orchestrator.git_ops import MergeResult

        config = OrchestratorConfig()
        req = MergeRequest(
            task_id='t-wire', branch=QueuedBranch.parse('task/t-wire', config.git.branch_prefix), worktree=tmp_path,
            pre_rebased=False, task_files=None, module_configs=[], config=config,
            result=asyncio.get_running_loop().create_future(),
        )
        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(
                success=True, merge_commit='deadbeef', merge_worktree=tmp_path,
            ),
            merge_wt=tmp_path,
            base_sha='items-own-base-sha',
            speculative=True,
            merged_branch_tip=None,
        )
        lease = HostLease(name='remote', runner=MagicMock(), is_local=False)

        captured: dict = {}

        async def _fake_run_post_merge_verify(*_args, **kwargs):
            captured.update(kwargs)
            return None  # pass

        worker = SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            _fake_run_post_merge_verify,
        ):
            await worker._run_inflight_verify(
                item, lease, depth=2, probe_base='deep-tip-commit',  # RED: no probe_base kwarg yet
            )

        assert captured.get('main_sha') == 'deep-tip-commit'
        # item itself must never be mutated by the probe_base override.
        assert item.base_sha == 'items-own-base-sha'
        # AMENDMENT (reviewer_comprehensive, task 2359): locks in the
        # label-vs-verified-content distinction documented on ProbePlacement
        # -- main_sha is attributed to the deep probed tip, but merge_sha
        # (the commit whose worktree is ACTUALLY built/verified, threaded
        # into the same merge_verify record) stays item's own merge_commit.
        # A firing probe never substitutes the deep tip's content for
        # item's own.
        assert captured.get('merge_sha') == 'deadbeef'

    async def test_no_probe_base_keeps_main_sha_byte_identical(self, tmp_path: Path) -> None:
        """probe_base=None (default / non-probed dispatch) -> main_sha is
        item.base_sha, unchanged from pre-task-2359 behaviour."""
        from orchestrator.git_ops import MergeResult

        config = OrchestratorConfig()
        req = MergeRequest(
            task_id='t-wire-default', branch=QueuedBranch.parse('task/t-wire-default', config.git.branch_prefix), worktree=tmp_path,
            pre_rebased=False, task_files=None, module_configs=[], config=config,
            result=asyncio.get_running_loop().create_future(),
        )
        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(
                success=True, merge_commit='deadbeef', merge_worktree=tmp_path,
            ),
            merge_wt=tmp_path,
            base_sha='items-own-base-sha',
            speculative=True,
            merged_branch_tip=None,
        )
        lease = HostLease(name='remote', runner=MagicMock(), is_local=False)

        captured: dict = {}

        async def _fake_run_post_merge_verify(*_args, **kwargs):
            captured.update(kwargs)
            return None  # pass

        worker = SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            _fake_run_post_merge_verify,
        ):
            await worker._run_inflight_verify(item, lease, depth=1)

        assert captured.get('main_sha') == 'items-own-base-sha'
