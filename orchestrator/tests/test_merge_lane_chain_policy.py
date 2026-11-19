"""How deep a merge verify chains (task 8528).

``orchestrator.merge_lane.chain_policy`` is the pure policy behind deep
merge-ahead chains (plans/merge-lane-policy-prd.md D7): depth comes from a
rolling per-item pass estimate over the newest tip verdicts of the rounds the
policy sized.
"""

from __future__ import annotations

import dataclasses

import pytest

from orchestrator.config import DepthRow
from orchestrator.merge_lane.chain_policy import (
    RECENT_WINDOW,
    ChainPolicyState,
    TipVerdict,
    choose_depth,
    on_tip_result,
    per_item_pass_estimate,
    seed_from_rows,
)
from orchestrator.merge_verify_facts import ChainDepthBasis, MergeVerifyDispatchFacts

_TABLE = [DepthRow(0.93, 8), DepthRow(0.85, 4), DepthRow(0.75, 2)]


def _verdicts(passes: int, total: int, *, depth: int = 1) -> tuple[TipVerdict, ...]:
    return tuple(TipVerdict(depth, index < passes) for index in range(total))


def _estimating(passes: int, total: int) -> ChainPolicyState:
    """A state whose per-item estimate is exactly ``passes / total``."""
    return ChainPolicyState(recent=_verdicts(passes, total))


class TestStateInvariants:
    def test_defaults_are_empty(self) -> None:
        assert ChainPolicyState().recent == ()

    def test_is_frozen(self) -> None:
        pytest.raises(
            dataclasses.FrozenInstanceError, setattr, ChainPolicyState(), 'recent', (),
        )

    def test_window_longer_than_recent_window_rejected(self) -> None:
        with pytest.raises(ValueError, match=str(RECENT_WINDOW)) as excinfo:
            ChainPolicyState(recent=_verdicts(RECENT_WINDOW + 1, RECENT_WINDOW + 1))
        assert str(RECENT_WINDOW + 1) in str(excinfo.value)

    def test_recent_window_is_thirty(self) -> None:
        assert RECENT_WINDOW == 30

    @pytest.mark.parametrize('bad_depth', [0, -1])
    def test_verdict_depth_below_one_rejected(self, bad_depth: int) -> None:
        with pytest.raises(ValueError, match='depth') as excinfo:
            ChainPolicyState(recent=(TipVerdict(bad_depth, True),))
        assert str(bad_depth) in str(excinfo.value)


class TestPerItemPassEstimate:
    def test_empty_window_has_no_estimate(self) -> None:
        assert per_item_pass_estimate(()) is None

    def test_pass_rate_is_spread_over_the_mean_depth(self) -> None:
        estimate = per_item_pass_estimate(_verdicts(27, 30, depth=2))
        assert estimate == pytest.approx(0.9 ** 0.5)

    def test_all_passes_is_one(self) -> None:
        assert per_item_pass_estimate(_verdicts(5, 5, depth=3)) == 1.0

    def test_all_fails_is_zero(self) -> None:
        assert per_item_pass_estimate(_verdicts(0, 5, depth=3)) == 0.0


class TestChooseDepth:
    @pytest.mark.parametrize(
        ('passes', 'total', 'depth', 'row'),
        [
            (19, 20, 8, (0.93, 8)),
            (9, 10, 4, (0.85, 4)),
            (4, 5, 2, (0.75, 2)),
            (7, 10, 1, None),
        ],
        ids=['0.95', '0.90', '0.80', '0.70'],
    )
    def test_first_row_the_estimate_meets(self, passes, total, depth, row) -> None:
        k, reason = choose_depth(_estimating(passes, total), 21, _TABLE)
        assert k == depth
        assert reason.basis is ChainDepthBasis.PASS_ESTIMATE
        assert reason.pass_estimate == pytest.approx(passes / total)
        assert reason.row == row
        assert reason.queue_depth == 21

    def test_a_threshold_is_met_at_equality(self) -> None:
        table = [DepthRow(0.9, 8), DepthRow(0.5, 2)]
        k, reason = choose_depth(_estimating(9, 10), 21, table)
        assert (k, reason.row) == (8, (0.9, 8))

    def test_a_short_queue_caps_the_depth(self) -> None:
        k, reason = choose_depth(_estimating(19, 20), 3, _TABLE)
        assert k == 3
        assert reason.queue_depth == 3
        assert reason.row == (0.93, 8)

    def test_nothing_but_the_queue_caps_the_depth(self) -> None:
        k, _ = choose_depth(_estimating(19, 20), 20, [DepthRow(0.5, 16)])
        assert k == 16

    def test_an_empty_window_takes_the_first_row(self) -> None:
        k, reason = choose_depth(ChainPolicyState(), 21, _TABLE)
        assert k == 8
        assert reason.basis is ChainDepthBasis.NO_ESTIMATE
        assert reason.pass_estimate is None
        assert reason.row == (0.93, 8)


class TestOnTipResult:
    @pytest.mark.parametrize('passed', [True, False])
    def test_a_verdict_joins_the_window(self, passed: bool) -> None:
        state = on_tip_result(ChainPolicyState(), 3, passed)
        assert state.recent == (TipVerdict(3, passed),)

    def test_the_window_keeps_the_newest_thirty(self) -> None:
        full = ChainPolicyState(recent=_verdicts(0, RECENT_WINDOW))
        state = on_tip_result(full, 2, True)
        assert len(state.recent) == RECENT_WINDOW
        assert state.recent[-1] == TipVerdict(2, True)
        assert state.recent[:-1] == full.recent[1:]

    def test_a_failed_chain_lowers_the_next_depth(self) -> None:
        state = ChainPolicyState(recent=_verdicts(19, 19))
        assert choose_depth(state, 21, _TABLE)[0] == 8
        for _ in range(2):
            state = on_tip_result(state, 1, False)
        k, reason = choose_depth(state, 21, _TABLE)
        assert (k, reason.row) == (4, (0.85, 4))


def _row(sha: str, passed: bool, **extra: object) -> dict[str, object]:
    return {'merge_sha': sha, 'passed': passed, 'attempt': 0, 'chain_items': 1, **extra}


class TestSeedFromRows:
    def test_rows_written_before_the_policy_seed_the_window(self) -> None:
        state = seed_from_rows([_row('s1', True, chain_items=3), _row('s2', False)])
        assert state.recent == (TipVerdict(3, True), TipVerdict(1, False))

    def test_a_null_reason_is_not_a_policy_sample(self) -> None:
        state = seed_from_rows([_row('s1', True, chain_depth_reason=None)])
        assert state.recent == ()

    def test_a_round_the_policy_did_not_size_does_not_seed(self) -> None:
        """A marked member's solo verify, a guard-declined round and a round
        under the kill switch all write the row an unsized dispatch writes."""
        unsized = MergeVerifyDispatchFacts().event_fields()
        assert 'chain_depth_reason' in unsized
        state = seed_from_rows([_row('s1', False, **unsized), _row('s2', True, **unsized)])
        assert state.recent == ()

    def test_only_the_last_attempt_per_merge_counts(self) -> None:
        state = seed_from_rows([
            _row('s1', False, chain_items=2),
            _row('s2', True),
            _row('s1', True, chain_items=2, attempt=1),
        ])
        assert state.recent == (TipVerdict(1, True), TipVerdict(2, True))

    def test_a_missing_chain_items_is_a_single_item(self) -> None:
        row = _row('s1', True)
        del row['chain_items']
        assert seed_from_rows([row]).recent == (TipVerdict(1, True),)

    def test_a_row_without_a_verdict_is_skipped(self) -> None:
        row = _row('s1', True)
        del row['passed']
        assert seed_from_rows([row]).recent == ()

    def test_the_newest_thirty_are_kept(self) -> None:
        rows = [_row(f's{index}', index >= 5) for index in range(RECENT_WINDOW + 5)]
        state = seed_from_rows(rows)
        assert len(state.recent) == RECENT_WINDOW
        assert all(verdict.passed for verdict in state.recent)


class TestOneMembershipRule:
    """The live fold and the start-up seed agree on which verdicts feed the estimate."""

    @pytest.mark.parametrize('passed', [True, False])
    def test_the_row_of_a_sized_round_seeds_the_verdict_its_live_fold_appends(
        self, passed: bool,
    ) -> None:
        _, reason = choose_depth(ChainPolicyState(), 3, _TABLE)
        sized = MergeVerifyDispatchFacts().with_chain_depth_reason(reason).event_fields()
        seeded = seed_from_rows([_row('s1', passed, chain_items=3, **sized)])
        assert seeded == on_tip_result(ChainPolicyState(), 3, passed)

    @pytest.mark.parametrize('basis', list(ChainDepthBasis))
    def test_every_basis_seeds(self, basis: ChainDepthBasis) -> None:
        reason = {'basis': basis.value, 'pass_estimate': None, 'row': None, 'queue_depth': 1}
        seeded = seed_from_rows([_row('s1', True, chain_depth_reason=reason)])
        assert seeded.recent == (TipVerdict(1, True),)

    def test_a_row_with_an_unknown_basis_is_skipped(self) -> None:
        reason = {'basis': 'halving', 'pass_estimate': None, 'row': None, 'queue_depth': 1}
        assert seed_from_rows([_row('s1', True, chain_depth_reason=reason)]).recent == ()
