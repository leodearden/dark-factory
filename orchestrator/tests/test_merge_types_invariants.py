"""I2 shape-validation invariants for the merge-queue message dataclasses (task 1990).

Retires the task-1928 field-drop bug class structurally: SpeculativeItem and
InflightEntry validate their own shape at construction time instead of relying
on every call site to hand-build a consistent set of fields.

Task ο (2000) supersedes the original XOR / merge_wt-iff-merge_result /
already_delivered ValueError tests: SpeculativeItem is now a
``RealMergeItem | DecidedItem`` union alias, so the illegal states those
tests used to construct-then-reject are UNREPRESENTABLE — attempting to pass
a cross-variant field is a TypeError (unknown kwarg) at construction time,
not a post-hoc ValueError from __post_init__. See
TestRealMergeItemUnrepresentability / TestDecidedItemUnrepresentability.

step-1  RED  — [ο] union unrepresentability (supersedes SpeculativeItem XOR / merge_wt-iff / already_delivered)
step-3  RED  — InflightEntry passthrough_outcome shadow invariant (retargeted to isinstance(item, DecidedItem))
step-7  RED  — InflightStatus str-compatible Enum
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.git_ops import MergeResult
from orchestrator.merge_queue import (
    DecidedItem,
    InflightEntry,
    InflightVerifyResult,
    MergeOutcome,
    RealMergeItem,
    SpeculativeItem,
)

# ── ο: RealMergeItem / DecidedItem unrepresentability (supersedes step-1 XOR/merge_wt/already_delivered) ──


def _real_kwargs() -> dict:
    """Minimal well-formed RealMergeItem kwargs."""
    return dict(
        request=MagicMock(),
        merge_result=MergeResult(success=True, merge_commit='deadbeef'),
        merge_wt=Path('/fake/merge-wt'),
        base_sha='aabbccdd',
        speculative=False,
    )


def _decided_kwargs() -> dict:
    """Minimal well-formed DecidedItem kwargs."""
    return dict(
        request=MagicMock(),
        base_sha='aabbccdd',
        speculative=False,
        immediate_outcome=MergeOutcome('blocked', reason='test'),
    )


class TestRealMergeItemUnrepresentability:
    """A RealMergeItem cannot carry any DECIDED-only field: passing one is a
    TypeError (unknown kwarg) at construction time, never a runtime shape
    check — the illegal state cannot be built at all."""

    def test_immediate_outcome_kwarg_raises_type_error(self) -> None:
        kwargs = _real_kwargs()
        kwargs['immediate_outcome'] = MergeOutcome('conflict')
        with pytest.raises(TypeError):
            RealMergeItem(**kwargs)

    def test_already_delivered_kwarg_raises_type_error(self) -> None:
        kwargs = _real_kwargs()
        kwargs['already_delivered'] = True
        with pytest.raises(TypeError):
            RealMergeItem(**kwargs)

    def test_lacks_decided_only_attributes(self) -> None:
        item = RealMergeItem(**_real_kwargs())
        assert not hasattr(item, 'immediate_outcome')
        assert not hasattr(item, 'already_delivered')
        assert not hasattr(item, 'failure_diagnostic')

    def test_real_shape_constructs(self) -> None:
        item = RealMergeItem(**_real_kwargs())
        assert item.merge_result is not None
        assert item.merge_wt is not None


class TestDecidedItemUnrepresentability:
    """A DecidedItem cannot carry any REAL-only field: passing one is a
    TypeError (unknown kwarg) at construction time, never a runtime shape
    check — the illegal state cannot be built at all."""

    def test_merge_result_kwarg_raises_type_error(self) -> None:
        kwargs = _decided_kwargs()
        kwargs['merge_result'] = MergeResult(success=True, merge_commit='deadbeef')
        with pytest.raises(TypeError):
            DecidedItem(**kwargs)

    def test_merge_wt_kwarg_raises_type_error(self) -> None:
        kwargs = _decided_kwargs()
        kwargs['merge_wt'] = Path('/fake/merge-wt')
        with pytest.raises(TypeError):
            DecidedItem(**kwargs)

    def test_lacks_real_only_attributes(self) -> None:
        item = DecidedItem(**_decided_kwargs())
        assert not hasattr(item, 'merge_result')
        assert not hasattr(item, 'merge_wt')
        assert not hasattr(item, 'merged_branch_tip')
        assert not hasattr(item, 'cap_permit')

    def test_decided_shape_constructs(self) -> None:
        item = DecidedItem(**_decided_kwargs())
        assert item.immediate_outcome is not None


class TestSpeculativeItemUnionAlias:
    """SpeculativeItem is a RealMergeItem | DecidedItem union alias,
    importable unchanged from both merge_types and the merge_queue shim."""

    def test_isinstance_real_against_union(self) -> None:
        assert isinstance(RealMergeItem(**_real_kwargs()), SpeculativeItem)

    def test_isinstance_decided_against_union(self) -> None:
        assert isinstance(DecidedItem(**_decided_kwargs()), SpeculativeItem)

    def test_importable_from_merge_types(self) -> None:
        from orchestrator.merge_types import DecidedItem as mt_DecidedItem
        from orchestrator.merge_types import RealMergeItem as mt_RealMergeItem
        from orchestrator.merge_types import SpeculativeItem as mt_SpeculativeItem
        assert mt_SpeculativeItem is SpeculativeItem
        assert mt_RealMergeItem is RealMergeItem
        assert mt_DecidedItem is DecidedItem


# ── step-3: InflightEntry passthrough_outcome shadow invariant ───────────────


def _entry_kwargs(item: SpeculativeItem, **overrides: object) -> dict:
    base: dict[str, object] = dict(
        item=item,
        lease=None,
        verify_task=None,
        merge_wt=None,
        was_speculative=False,
        phase='passthrough',
        passthrough_outcome=None,
    )
    base.update(overrides)
    return base


class TestInflightEntryPassthroughInvariant:
    """passthrough_outcome is not None => item is a DecidedItem."""

    def test_passthrough_outcome_wrapping_real_item_raises(self) -> None:
        real_item = RealMergeItem(**_real_kwargs())
        with pytest.raises(ValueError):
            InflightEntry(**_entry_kwargs(real_item, passthrough_outcome=MergeOutcome('conflict')))

    def test_passthrough_outcome_wrapping_decided_item_constructs(self) -> None:
        decided_item = DecidedItem(**_decided_kwargs())
        entry = InflightEntry(
            **_entry_kwargs(decided_item, passthrough_outcome=decided_item.immediate_outcome),
        )
        assert entry.passthrough_outcome is decided_item.immediate_outcome

    def test_no_passthrough_outcome_wrapping_any_item_constructs(self) -> None:
        real_item = RealMergeItem(**_real_kwargs())
        decided_item = DecidedItem(**_decided_kwargs())
        InflightEntry(**_entry_kwargs(real_item, passthrough_outcome=None))
        InflightEntry(**_entry_kwargs(decided_item, passthrough_outcome=None))


# ── step-7: InflightStatus str-compatible Enum ───────────────────────────────


class TestInflightStatusEnum:
    """InflightStatus is a single str-compatible Enum shared by
    InflightEntry.status and InflightVerifyResult.status.

    Imported inside each test body (not at module scope) so that, until
    step-8 lands the enum, only these tests fail with ImportError — the
    rest of this file's already-green step-1/step-3 tests keep collecting
    and passing.
    """

    _MEMBER_NAMES = (
        'DROPPED',
        'REQUEUED',
        'RUNNER_UNAVAILABLE',
        'ABANDONED_PREDISPATCH',
        'REQUEUED_PREDISPATCH',
    )

    def test_importable_from_merge_types(self) -> None:
        from orchestrator.merge_types import InflightStatus
        assert issubclass(InflightStatus, str)

    def test_importable_from_merge_queue_reexport_shim(self) -> None:
        from orchestrator.merge_queue import InflightStatus
        assert issubclass(InflightStatus, str)

    def test_members_exist_with_value_equal_to_name(self) -> None:
        from orchestrator.merge_types import InflightStatus
        for name in self._MEMBER_NAMES:
            member = getattr(InflightStatus, name)
            assert member.value == name

    def test_str_compatibility(self) -> None:
        from orchestrator.merge_types import InflightStatus
        assert InflightStatus.DROPPED == 'DROPPED'
        assert InflightStatus.REQUEUED in ('DROPPED', 'REQUEUED')

    def test_inflight_verify_result_status_is_str_compatible(self) -> None:
        from orchestrator.merge_types import InflightStatus
        result = InflightVerifyResult(outcome=None, merge_wt=None, status=InflightStatus.DROPPED)
        assert result.status == 'DROPPED'


# ── amendment (task 1990 review): InflightVerifyResult status subset ────────


class TestInflightVerifyResultStatusInvariant:
    """InflightVerifyResult.status is restricted to the 3 verify-result
    sentinels; the 2 predispatch-only sentinels belong exclusively to
    InflightEntry and must be rejected here (parity with SpeculativeItem /
    InflightEntry, which both validate their shape in __post_init__)."""

    def test_none_constructs(self) -> None:
        result = InflightVerifyResult(outcome=None, merge_wt=None, status=None)
        assert result.status is None

    def test_each_valid_member_constructs(self) -> None:
        from orchestrator.merge_types import InflightStatus
        for member in (
            InflightStatus.DROPPED,
            InflightStatus.REQUEUED,
            InflightStatus.RUNNER_UNAVAILABLE,
        ):
            result = InflightVerifyResult(outcome=None, merge_wt=None, status=member)
            assert result.status == member

    def test_raw_string_equal_to_valid_member_constructs(self) -> None:
        # Matches existing test fixtures (e.g. test_merge_queue_concurrent_verify.py's
        # _make_sentinel_entry) that pass a raw string rather than the enum member;
        # InflightStatus's str-compatibility must keep this working at runtime even
        # though a bare literal is a static type mismatch (deliberately exercised here).
        result = InflightVerifyResult(outcome=None, merge_wt=None, status='DROPPED')  # type: ignore[arg-type]
        assert result.status == 'DROPPED'

    def test_predispatch_only_member_raises(self) -> None:
        from orchestrator.merge_types import InflightStatus
        with pytest.raises(ValueError):
            InflightVerifyResult(
                outcome=None, merge_wt=None, status=InflightStatus.ABANDONED_PREDISPATCH,
            )

    def test_other_predispatch_only_member_raises(self) -> None:
        from orchestrator.merge_types import InflightStatus
        with pytest.raises(ValueError):
            InflightVerifyResult(
                outcome=None, merge_wt=None, status=InflightStatus.REQUEUED_PREDISPATCH,
            )
