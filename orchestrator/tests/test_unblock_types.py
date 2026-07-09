"""Tests for orchestrator.unblock_types — typed BlockClass/BlockRecord bridge
(PRD: plans/verify-plan-prd.md task ζ, Contract §BlockRecord, Open Q5).

Dual-read bridge that lets b3_gate.check_proposal branch on a typed
``block_class`` field instead of a fragile ``block_reason`` prefix sniff and
a ``status``-key-presence sniff. See b3_gate.py / dry_run_unblock.py /
workflow.py for producer/consumer wiring (steps 3-8); this module covers the
core surface only:

  step-1: BlockClass member set + StrEnum wire byte-identity (mirrors task
          α's FailureCategory pattern in verify_categories.py);
          classify_block_reason totality; BlockRecord round-trip
          invariant B1.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

# ---------------------------------------------------------------------------
# step-1: BlockClass member set + StrEnum wire byte-identity
# ---------------------------------------------------------------------------


class TestBlockClassMemberSet:
    """BlockClass is a StrEnum with exactly the 4 named semantic members
    (Open Q5: a total reason-prefix->class mapping onto 4 semantic classes,
    NOT one member per raw block-reason prefix)."""

    def test_member_values_match_expected_set_exactly(self):
        from orchestrator.unblock_types import BlockClass
        assert {c.value for c in BlockClass} == {
            'agent_failure', 'review_issues', 'merge_verify_red', 'post_merge_red_main',
        }

    def test_member_count_is_four(self):
        from orchestrator.unblock_types import BlockClass
        assert len(list(BlockClass)) == 4

    def test_is_strenum_subclass(self):
        from enum import StrEnum

        from orchestrator.unblock_types import BlockClass
        assert issubclass(BlockClass, StrEnum)

    def test_named_members_have_expected_values(self):
        from orchestrator.unblock_types import BlockClass
        assert BlockClass.AGENT_FAILURE == 'agent_failure'
        assert BlockClass.REVIEW_ISSUES == 'review_issues'
        assert BlockClass.MERGE_VERIFY_RED == 'merge_verify_red'
        assert BlockClass.POST_MERGE_RED_MAIN == 'post_merge_red_main'


class TestBlockClassWireByteIdentity:
    """Every member IS its str value — json.dumps output is unchanged
    (mirrors verify_categories.FailureCategory's F2 wire-identity guard)."""

    def test_str_of_every_member_equals_its_value(self):
        from orchestrator.unblock_types import BlockClass
        for member in BlockClass:
            assert str(member) == member.value

    def test_json_dumps_of_every_member_matches_plain_str(self):
        from orchestrator.unblock_types import BlockClass
        for member in BlockClass:
            assert json.dumps({'block_class': member}) == json.dumps({'block_class': member.value})

    def test_member_equals_and_hashes_as_plain_str(self):
        from orchestrator.unblock_types import BlockClass
        for member in BlockClass:
            assert member == member.value
            assert hash(member) == hash(member.value)


# ---------------------------------------------------------------------------
# step-1: classify_block_reason — total prefix -> class mapping
# ---------------------------------------------------------------------------


class TestClassifyBlockReason:
    """classify_block_reason is a TOTAL function: every str reason maps to
    exactly one BlockClass, defaulting to AGENT_FAILURE for anything
    unrecognized (totality — Open Q5 resolution)."""

    def test_post_merge_pyright_prefix_maps_to_post_merge_red_main(self):
        from orchestrator.merge_gates import POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX
        from orchestrator.unblock_types import BlockClass, classify_block_reason
        reason = (
            POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX
            + ': post-merge unscoped type-check failed for orchestrator'
        )
        assert classify_block_reason(reason) == BlockClass.POST_MERGE_RED_MAIN

    @pytest.mark.parametrize('prefix_name', [
        'DROPPED_PLAN_TARGETS_REASON_PREFIX',
        'POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX',
        'PLAN_FILES_NOT_TOUCHED_REASON_PREFIX',
    ])
    def test_merge_verify_prefixes_map_to_merge_verify_red(self, prefix_name):
        from orchestrator import merge_gates
        from orchestrator.unblock_types import BlockClass, classify_block_reason
        prefix = getattr(merge_gates, prefix_name)
        assert classify_block_reason(f'{prefix}: some detail') == BlockClass.MERGE_VERIFY_RED

    @pytest.mark.parametrize('reason', [
        'Workflow error: boom',
        'Planning failed: bad plan',
        'verify exhausted',
        '',
    ])
    def test_unrecognized_reasons_default_to_agent_failure(self, reason):
        from orchestrator.unblock_types import BlockClass, classify_block_reason
        assert classify_block_reason(reason) == BlockClass.AGENT_FAILURE

    def test_exact_prefix_with_no_suffix_still_matches(self):
        # startswith semantics: the bare prefix (no ': detail' suffix) must
        # still classify correctly — proves this is a prefix match, not an
        # exact-template match.
        from orchestrator.merge_gates import POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX
        from orchestrator.unblock_types import BlockClass, classify_block_reason
        assert (
            classify_block_reason(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX)
            == BlockClass.POST_MERGE_RED_MAIN
        )

    def test_return_type_is_always_block_class(self):
        from orchestrator.unblock_types import BlockClass, classify_block_reason
        assert isinstance(classify_block_reason('anything at all'), BlockClass)


# ---------------------------------------------------------------------------
# step-1: BlockRecord — invariant B1 (round-trip) + to_dict/from_dict shape
# ---------------------------------------------------------------------------


class TestBlockRecordRoundTrip:
    """Invariant B1: BlockRecord.from_dict(BlockRecord.to_dict(r)) == r."""

    def test_round_trip_full_record(self):
        from orchestrator.unblock_types import BlockClass, BlockRecord
        record = BlockRecord(
            block_class=BlockClass.MERGE_VERIFY_RED,
            risk_label='low',
            head_sha='aaabbbccc',
            main_sha='dddeeefff',
            files_referenced=['foo.py', 'bar.py'],
            investigated_at='2026-06-04T09:00:00+00:00',
        )
        assert BlockRecord.from_dict(record.to_dict()) == record

    def test_round_trip_none_shas_and_empty_files(self):
        from orchestrator.unblock_types import BlockClass, BlockRecord
        record = BlockRecord(
            block_class=BlockClass.AGENT_FAILURE,
            risk_label='human-review-required',
            head_sha=None,
            main_sha=None,
            files_referenced=[],
            investigated_at='',
        )
        assert BlockRecord.from_dict(record.to_dict()) == record

    def test_to_dict_block_class_is_plain_str_value(self):
        # json-serialisable: to_dict()['block_class'] must be the plain str
        # value, not the StrEnum member (though StrEnum IS a str, this pins
        # the wire shape explicitly).
        from orchestrator.unblock_types import BlockClass, BlockRecord
        record = BlockRecord(
            block_class=BlockClass.POST_MERGE_RED_MAIN,
            risk_label='low',
            head_sha='a',
            main_sha='b',
            files_referenced=[],
            investigated_at='t',
        )
        d = record.to_dict()
        assert d['block_class'] == 'post_merge_red_main'
        assert type(d['block_class']) is str
        assert json.dumps(d)  # must be fully json-serialisable

    def test_from_dict_tolerates_superset_dict(self):
        # A real proposal entry carries many sibling keys (proposal_text,
        # status, block_reason, timestamp, cost_usd, diagnostics...) —
        # from_dict must read exactly the 6 BlockRecord keys and ignore
        # everything else.
        from orchestrator.unblock_types import BlockClass, BlockRecord
        entry = {
            'proposal_text': 'Fix it',
            'status': 'investigation_failed',
            'block_reason': 'verify exhausted',
            'timestamp': '2026-06-04T09:00:00+00:00',
            'cost_usd': 0.5,
            'block_class': 'agent_failure',
            'risk_label': 'human-review-required',
            'head_sha': 'aaabbbccc',
            'main_sha': 'dddeeefff',
            'files_referenced': ['foo.py'],
            'investigated_at': '2026-06-04T09:00:00+00:00',
        }
        record = BlockRecord.from_dict(entry)
        assert record == BlockRecord(
            block_class=BlockClass.AGENT_FAILURE,
            risk_label='human-review-required',
            head_sha='aaabbbccc',
            main_sha='dddeeefff',
            files_referenced=['foo.py'],
            investigated_at='2026-06-04T09:00:00+00:00',
        )

    def test_is_frozen_dataclass(self):
        from orchestrator.unblock_types import BlockClass, BlockRecord
        assert dataclasses.is_dataclass(BlockRecord)
        record = BlockRecord(
            block_class=BlockClass.AGENT_FAILURE,
            risk_label='low',
            head_sha=None,
            main_sha=None,
            files_referenced=[],
            investigated_at='',
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.risk_label = 'medium'  # type: ignore[misc]

    def test_has_expected_fields(self):
        from orchestrator.unblock_types import BlockRecord
        field_names = {f.name for f in dataclasses.fields(BlockRecord)}
        assert field_names == {
            'block_class', 'risk_label', 'head_sha', 'main_sha',
            'files_referenced', 'investigated_at',
        }
