"""The typed dispatch facts a ``merge_verify`` row carries.

``orchestrator.merge_verify_facts`` is the closed wire vocabulary the
merge-lane-policy PRD (``plans/merge-lane-policy-prd.md`` D10) keys on:
``runner_reason`` says why a verify ran on the host it ran on,
``base_relation`` says what the verified merge was built on, and
``chain_depth_reason`` says why the deep policy chose the verify's depth.
"""
from __future__ import annotations

import dataclasses

import pytest

from orchestrator.merge_verify_facts import (
    BaseRelation,
    ChainDepthBasis,
    ChainDepthReason,
    ChainMembers,
    MergeVerifyDispatchFacts,
    RunnerReason,
    classify_base_relation,
)


class TestClassifyBaseRelation:
    def test_base_equal_to_main_is_main(self) -> None:
        assert classify_base_relation(
            'm1', main_sha='m1', stacked_on=set(), rebased_heads=set(),
        ) is BaseRelation.MAIN

    def test_base_on_a_rebased_head_is_rebased_head(self) -> None:
        assert classify_base_relation(
            'r1', main_sha='m1', stacked_on=set(), rebased_heads={'r1'},
        ) is BaseRelation.REBASED_HEAD

    def test_rebased_head_is_checked_before_stacked(self) -> None:
        assert classify_base_relation(
            'x1', main_sha='m1', stacked_on={'x1'}, rebased_heads={'x1'},
        ) is BaseRelation.REBASED_HEAD

    def test_base_on_an_inflight_merge_commit_is_stacked(self) -> None:
        assert classify_base_relation(
            's1', main_sha='m1', stacked_on={'s1'}, rebased_heads=set(),
        ) is BaseRelation.STACKED

    def test_base_off_main_and_off_the_frontier_is_stale(self) -> None:
        assert classify_base_relation(
            'old', main_sha='m1', stacked_on={'s1'}, rebased_heads={'r1'},
        ) is BaseRelation.STALE

    def test_unreadable_main_and_unknown_base_is_unknown(self) -> None:
        assert classify_base_relation(
            'old', main_sha=None, stacked_on={'s1'}, rebased_heads={'r1'},
        ) is None

    def test_unreadable_main_still_classifies_a_stacked_base(self) -> None:
        assert classify_base_relation(
            's1', main_sha=None, stacked_on={'s1'}, rebased_heads=set(),
        ) is BaseRelation.STACKED

    def test_whitespace_around_shas_is_stripped(self) -> None:
        assert classify_base_relation(
            ' m1\n', main_sha='m1\n', stacked_on=set(), rebased_heads=set(),
        ) is BaseRelation.MAIN
        assert classify_base_relation(
            's1\n', main_sha='m1', stacked_on={' s1 '}, rebased_heads=set(),
        ) is BaseRelation.STACKED
        assert classify_base_relation(
            'r1 ', main_sha='m1', stacked_on=set(), rebased_heads={'\nr1'},
        ) is BaseRelation.REBASED_HEAD


class TestMergeVerifyDispatchFacts:
    def test_event_fields_are_plain_strings(self) -> None:
        facts = MergeVerifyDispatchFacts(
            runner_reason=RunnerReason.REMOTE_BUSY,
            base_relation=BaseRelation.STACKED,
        )
        fields = facts.event_fields()
        assert fields == {
            'runner_reason': 'remote_busy', 'base_relation': 'stacked',
            'chain_depth_reason': None,
        }
        assert type(fields['runner_reason']) is str
        assert type(fields['base_relation']) is str

    def test_event_fields_of_no_facts_are_present_and_null(self) -> None:
        assert MergeVerifyDispatchFacts().event_fields() == {
            'runner_reason': None, 'base_relation': None, 'chain_depth_reason': None,
        }

    def test_as_retry_marks_the_held_host_and_keeps_the_base(self) -> None:
        original = MergeVerifyDispatchFacts(
            runner_reason=RunnerReason.PREFER_REMOTE_FREE,
            base_relation=BaseRelation.MAIN,
        )
        retry = original.as_retry()
        assert retry is not original
        assert retry.runner_reason is RunnerReason.RETRY_HELD_HOST
        assert retry.base_relation is BaseRelation.MAIN
        assert original.runner_reason is RunnerReason.PREFER_REMOTE_FREE

    def test_local_only_has_no_base_relation(self) -> None:
        facts = MergeVerifyDispatchFacts.local_only()
        assert facts.runner_reason is RunnerReason.LOCAL_ONLY_RULE
        assert facts.base_relation is None

    def test_is_frozen(self) -> None:
        pytest.raises(
            dataclasses.FrozenInstanceError, setattr,
            MergeVerifyDispatchFacts(), 'runner_reason', RunnerReason.LOCAL_BUSY,
        )


def _table_row_reason() -> ChainDepthReason:
    return ChainDepthReason(
        basis=ChainDepthBasis.PASS_ESTIMATE,
        pass_estimate=0.95,
        row=(0.93, 8),
        queue_depth=12,
    )


class TestChainDepthReason:
    def test_wire_of_a_table_row_choice(self) -> None:
        wire = _table_row_reason().wire()
        assert wire == {
            'basis': 'pass_estimate',
            'pass_estimate': 0.95,
            'row': [0.93, 8],
            'queue_depth': 12,
        }
        assert type(wire['basis']) is str

    def test_wire_of_a_cold_window_has_no_estimate(self) -> None:
        reason = ChainDepthReason(
            basis=ChainDepthBasis.NO_ESTIMATE,
            pass_estimate=None,
            row=(0.93, 8),
            queue_depth=3,
        )
        assert reason.wire() == {
            'basis': 'no_estimate',
            'pass_estimate': None,
            'row': [0.93, 8],
            'queue_depth': 3,
        }

    def test_wire_of_an_estimate_below_every_row_has_no_row(self) -> None:
        reason = ChainDepthReason(
            basis=ChainDepthBasis.PASS_ESTIMATE,
            pass_estimate=0.5,
            row=None,
            queue_depth=5,
        )
        assert reason.wire() == {
            'basis': 'pass_estimate',
            'pass_estimate': 0.5,
            'row': None,
            'queue_depth': 5,
        }

    def test_is_frozen(self) -> None:
        pytest.raises(
            dataclasses.FrozenInstanceError, setattr,
            _table_row_reason(), 'queue_depth', 1,
        )


class TestChainDepthReasonOnTheRow:
    def test_with_chain_depth_reason_is_a_new_value(self) -> None:
        original = MergeVerifyDispatchFacts(runner_reason=RunnerReason.REMOTE_BUSY)
        reasoned = original.with_chain_depth_reason(_table_row_reason())
        assert reasoned is not original
        assert original.chain_depth_reason is None
        assert reasoned.chain_depth_reason == _table_row_reason()
        assert reasoned.runner_reason is RunnerReason.REMOTE_BUSY

    def test_event_fields_carry_the_wire_shape(self) -> None:
        facts = MergeVerifyDispatchFacts().with_chain_depth_reason(_table_row_reason())
        assert facts.event_fields()['chain_depth_reason'] == _table_row_reason().wire()

    def test_reason_survives_covering_and_retry(self) -> None:
        members = ChainMembers(('mr-a', 'mr-b'), ('101', '102'))
        facts = (
            MergeVerifyDispatchFacts(runner_reason=RunnerReason.PREFER_REMOTE_FREE)
            .with_chain_depth_reason(_table_row_reason())
            .covering(members)
            .as_retry()
        )
        assert facts.chain_depth_reason == _table_row_reason()
        assert facts.members == members
        assert facts.event_fields()['chain_depth_reason'] == _table_row_reason().wire()


class TestClosedVocabulary:
    def test_runner_reason_values(self) -> None:
        assert {r.value for r in RunnerReason} == {
            'prefer_remote_free', 'remote_busy', 'remote_quarantined',
            'local_only_rule', 'gate_on_head_host', 'retry_held_host',
            'prefer_local_free', 'local_busy',
        }

    def test_base_relation_values(self) -> None:
        assert {b.value for b in BaseRelation} == {
            'main', 'stacked', 'rebased_head', 'stale',
        }

    def test_chain_depth_basis_values(self) -> None:
        assert {b.value for b in ChainDepthBasis} == {
            'pass_estimate', 'no_estimate',
        }
