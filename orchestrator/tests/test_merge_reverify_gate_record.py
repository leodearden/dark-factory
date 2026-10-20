"""The post-rebase re-verify gate records every decision it makes (task 5064).

``gates.py::_reverify_rebased_tree`` decides whether a rebased tree may
advance without a fresh verify. Each decision, skip or fire, is a
``merge_reverify_gate`` row, and a fire's re-verify writes its own
``merge_verify`` row beside it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.event_store import EventStore, EventType
from orchestrator.merge_lane.reverify_gate_decision import (
    MAX_RECORDED_OVERLAP_PATHS,
    GateClearance,
    ReverifyGateDecision,
    ReverifyGateReason,
    emit_reverify_gate_decision,
)

_SHAS = {
    'merge_sha': 'a' * 40,
    'verified_merge_sha': 'b' * 40,
    'rebased_from': 'c' * 40,
    'rebased_onto': 'd' * 40,
}


def _decision(
    reason: ReverifyGateReason,
    *,
    overlap: tuple[str, ...] = (),
    blockers: tuple[str, ...] = (),
    task_id: str = 't-5064',
) -> ReverifyGateDecision:
    return ReverifyGateDecision(
        task_id=task_id, reason=reason,
        merge_sha=_SHAS['merge_sha'],
        verified_merge_sha=_SHAS['verified_merge_sha'],
        rebased_from=_SHAS['rebased_from'],
        rebased_onto=_SHAS['rebased_onto'],
        overlap=overlap, blockers=blockers,
    )


_CONSISTENT_DECISIONS = {
    ReverifyGateReason.same_tree: _decision(ReverifyGateReason.same_tree),
    ReverifyGateReason.disjoint: _decision(ReverifyGateReason.disjoint),
    ReverifyGateReason.disjoint_untrusted: _decision(
        ReverifyGateReason.disjoint_untrusted,
        blockers=('whole-tree merge gate (merge_verify_breadth=full)',),
    ),
    ReverifyGateReason.overlap: _decision(ReverifyGateReason.overlap, overlap=('x.txt',)),
    ReverifyGateReason.overlap_unknown: _decision(ReverifyGateReason.overlap_unknown),
}

_SKIPS = {ReverifyGateReason.same_tree, ReverifyGateReason.disjoint}


class TestReverifyGateDecisionRecord:
    def test_every_reason_has_a_consistent_decision_under_test(self) -> None:
        assert set(_CONSISTENT_DECISIONS) == set(ReverifyGateReason)

    @pytest.mark.parametrize('reason', list(ReverifyGateReason), ids=str)
    def test_event_data_carries_the_decision(self, reason: ReverifyGateReason) -> None:
        decision = _CONSISTENT_DECISIONS[reason]

        assert decision.event_data() == {
            'reason': str(reason),
            'reverify': reason not in _SKIPS,
            **_SHAS,
            'overlap': list(decision.overlap),
            'overlap_total': len(decision.overlap),
            'blockers': list(decision.blockers),
        }

    def test_recorded_overlap_is_bounded_and_says_so(self) -> None:
        paths = tuple(f'f{n:02d}.py' for n in range(MAX_RECORDED_OVERLAP_PATHS + 2))
        decision = _decision(ReverifyGateReason.overlap, overlap=paths)

        data = decision.event_data()

        assert data['overlap'] == list(paths[:MAX_RECORDED_OVERLAP_PATHS])
        assert data['overlap_total'] == len(paths)

    @pytest.mark.parametrize(
        ('reason', 'overlap', 'blockers'),
        [
            pytest.param(ReverifyGateReason.overlap, (), (), id='overlap-without-paths'),
            pytest.param(ReverifyGateReason.disjoint, ('x.txt',), (), id='disjoint-with-paths'),
            pytest.param(
                ReverifyGateReason.disjoint_untrusted, (), (), id='untrusted-without-blockers',
            ),
            pytest.param(
                ReverifyGateReason.overlap, ('x.txt',), ('unverified drift',),
                id='overlap-with-blockers',
            ),
        ],
    )
    def test_an_inconsistent_decision_is_refused_by_name(
        self, reason: ReverifyGateReason, overlap: tuple[str, ...], blockers: tuple[str, ...],
    ) -> None:
        with pytest.raises(ValueError) as refused:
            _decision(reason, overlap=overlap, blockers=blockers, task_id='t-refused')

        message = str(refused.value)
        for fact in ('t-refused', str(reason), repr(overlap), repr(blockers)):
            assert fact in message, message

    @pytest.mark.parametrize('reason', list(ReverifyGateReason), ids=str)
    def test_only_a_skip_yields_a_clearance(self, reason: ReverifyGateReason) -> None:
        expected = {
            ReverifyGateReason.same_tree: GateClearance.same_tree,
            ReverifyGateReason.disjoint: GateClearance.disjoint,
        }.get(reason)

        assert reason.skip_clearance is expected
        assert reason.reverifies is (expected is None)
        assert _CONSISTENT_DECISIONS[reason].reverifies is reason.reverifies

    def test_emit_writes_one_runs_db_row(self, tmp_path: Path) -> None:
        store = EventStore(tmp_path / 'runs.db', 'run-5064')
        decision = _CONSISTENT_DECISIONS[ReverifyGateReason.overlap]

        emit_reverify_gate_decision(store, decision)

        [row] = store.fetch_events_by_type(EventType.merge_reverify_gate)
        assert row['task_id'] == decision.task_id
        assert row['phase'] == 'merge'
        assert row['data'] == decision.event_data()

    def test_emit_without_a_store_is_a_no_op(self) -> None:
        emit_reverify_gate_decision(None, _CONSISTENT_DECISIONS[ReverifyGateReason.disjoint])
