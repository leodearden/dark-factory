"""I2 shape-validation invariants for the merge-queue message dataclasses (task 1990).

Retires the task-1928 field-drop bug class structurally: SpeculativeItem and
InflightEntry validate their own shape at construction time instead of relying
on every call site to hand-build a consistent set of fields.

step-1  RED  — SpeculativeItem XOR / merge_wt-iff-merge_result / already_delivered
step-3  RED  — InflightEntry passthrough_outcome shadow invariant
step-7  RED  — InflightStatus str-compatible Enum
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.git_ops import MergeResult
from orchestrator.merge_queue import (
    InflightEntry,
    InflightVerifyResult,
    MergeOutcome,
    SpeculativeItem,
)

# ── step-1: SpeculativeItem shape validation ─────────────────────────────────


def _real_kwargs() -> dict:
    """Minimal well-formed REAL SpeculativeItem kwargs (merge_result+merge_wt set)."""
    return dict(
        request=MagicMock(),
        merge_result=MergeResult(success=True, merge_commit='deadbeef'),
        merge_wt=Path('/fake/merge-wt'),
        base_sha='aabbccdd',
        speculative=False,
        skip_verify=False,
    )


def _decided_kwargs() -> dict:
    """Minimal well-formed DECIDED SpeculativeItem kwargs (immediate_outcome set)."""
    return dict(
        request=MagicMock(),
        merge_result=None,
        merge_wt=None,
        base_sha='aabbccdd',
        speculative=False,
        skip_verify=False,
        immediate_outcome=MergeOutcome('blocked', reason='test'),
    )


class TestSpeculativeItemXorInvariant:
    """Exactly one of {merge_result, immediate_outcome} may be non-None."""

    def test_both_set_raises(self) -> None:
        kwargs = _real_kwargs()
        kwargs['immediate_outcome'] = MergeOutcome('conflict')
        with pytest.raises(ValueError):
            SpeculativeItem(**kwargs)

    def test_both_none_raises(self) -> None:
        kwargs = _real_kwargs()
        kwargs['merge_result'] = None
        kwargs['merge_wt'] = None
        with pytest.raises(ValueError):
            SpeculativeItem(**kwargs)

    def test_real_shape_constructs(self) -> None:
        item = SpeculativeItem(**_real_kwargs())
        assert item.merge_result is not None
        assert item.immediate_outcome is None

    def test_decided_shape_constructs(self) -> None:
        item = SpeculativeItem(**_decided_kwargs())
        assert item.immediate_outcome is not None
        assert item.merge_result is None


class TestSpeculativeItemMergeWtInvariant:
    """merge_wt is not None iff merge_result is not None."""

    def test_merge_result_without_merge_wt_raises(self) -> None:
        kwargs = _real_kwargs()
        kwargs['merge_wt'] = None
        with pytest.raises(ValueError):
            SpeculativeItem(**kwargs)

    def test_merge_wt_without_merge_result_raises(self) -> None:
        kwargs = _decided_kwargs()
        kwargs['immediate_outcome'] = None
        kwargs['merge_wt'] = Path('/fake/merge-wt')
        with pytest.raises(ValueError):
            SpeculativeItem(**kwargs)


class TestSpeculativeItemAlreadyDeliveredInvariant:
    """already_delivered=True requires immediate_outcome to be set."""

    def test_already_delivered_without_immediate_outcome_raises(self) -> None:
        kwargs = _real_kwargs()
        kwargs['already_delivered'] = True
        with pytest.raises(ValueError):
            SpeculativeItem(**kwargs)

    def test_already_delivered_with_immediate_outcome_constructs(self) -> None:
        kwargs = _decided_kwargs()
        kwargs['already_delivered'] = True
        item = SpeculativeItem(**kwargs)
        assert item.already_delivered is True


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
    """passthrough_outcome is not None => item.immediate_outcome is not None."""

    def test_passthrough_outcome_wrapping_real_item_raises(self) -> None:
        real_item = SpeculativeItem(**_real_kwargs())
        with pytest.raises(ValueError):
            InflightEntry(**_entry_kwargs(real_item, passthrough_outcome=MergeOutcome('conflict')))

    def test_passthrough_outcome_wrapping_decided_item_constructs(self) -> None:
        decided_item = SpeculativeItem(**_decided_kwargs())
        entry = InflightEntry(
            **_entry_kwargs(decided_item, passthrough_outcome=decided_item.immediate_outcome),
        )
        assert entry.passthrough_outcome is decided_item.immediate_outcome

    def test_no_passthrough_outcome_wrapping_any_item_constructs(self) -> None:
        real_item = SpeculativeItem(**_real_kwargs())
        decided_item = SpeculativeItem(**_decided_kwargs())
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
